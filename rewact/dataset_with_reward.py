import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union


class LeRobotDatasetWithReward:
    """
    A wrapper around LeRobotDataset that adds reward calculation with support for user-defined keypoints.
    
    This class enhances LeRobotDataset by adding reward information to each frame. Rewards can be calculated in two ways:
    
    1. **Keypoint-based interpolation** (preferred): Users can define specific reward values at specific frames
       within episodes. The class automatically interpolates between these keypoints to provide smooth reward
       trajectories. Special handling is included for sharp transitions (e.g., episode resets from success to start).
       
    2. **Fallback linear interpolation**: When no keypoints are defined for an episode, rewards are calculated
       using a simple linear interpolation between reward_start_pct and reward_end_pct based on episode progress.
    
    Key features:
    - Automatic saving/loading of keypoint rewards to/from JSON files in the dataset's meta directory
    - Smooth interpolation between keypoints with support for sharp transitions
    - Caching of interpolated rewards for performance
    - Validation of reward values (must be between 0.0 and 1.0)
    - Non-intrusive: works alongside existing LeRobot datasets without modification
    
    Example usage:
        ```python
        # Load dataset and wrap with reward functionality
        dataset = LeRobotDataset("my_dataset")
        reward_dataset = LeRobotDatasetWithReward(dataset)
        
        # Add keypoint rewards for episode 3
        episode_rewards = {
            5: 0.0,      # frame 5: start
            87: 0.33,    # frame 87: first milestone
            104: 0.66,   # frame 104: second milestone  
            160: 1.0     # frame 160: success
        }
        reward_dataset.add_episode_rewards(3, episode_rewards)
        
        # Access data with interpolated rewards
        item = reward_dataset[idx]
        reward = item["reward"]  # Tensor with interpolated reward value
        ```
    """
    
    def __init__(
        self,
        dataset,
        reward_start_pct: float = 0.05,
        reward_end_pct: float = 0.95,
    ):
        """
        Initialize the dataset wrapper with reward calculation parameters.
        
        Args:
            dataset: The underlying LeRobotDataset
            reward_start_pct: Progress percentage where reward starts (0.0) - used for fallback
            reward_end_pct: Progress percentage where reward reaches maximum (1.0) - used for fallback
        """
        # Create the underlying dataset
        self._dataset = dataset

        # Validate reward parameters
        if not (0.0 <= reward_start_pct <= 1.0 and 0.0 <= reward_end_pct <= 1.0):
            raise ValueError("Reward percentages must be between 0.0 and 1.0")
        if reward_start_pct >= reward_end_pct:
            raise ValueError("reward_start_pct must be less than reward_end_pct")

        # Store reward parameters (fallback when no keypoints are defined)
        self.reward_start_pct = reward_start_pct
        self.reward_end_pct = reward_end_pct
        
        # Load keypoint rewards if they exist
        self.keypoint_rewards = self._load_keypoint_rewards()
        self._episode_reward_cache = {}


    def __getitem__(self, idx) -> dict:
        """Get an item from the dataset with calculated reward."""
        item = self._dataset[idx]
        
        # Add reward calculation
        item["reward"] = self._calculate_reward(idx, item)
        
        return item
    
    def _calculate_reward(self, idx: int, item: dict) -> torch.Tensor:
        """
        Calculate reward based on keypoint interpolation or fallback linear interpolation.
        
        Args:
            idx: Global frame index
            item: Data item from the dataset
            
        Returns:
            Reward tensor (float32)
        """
        ep_idx = item["episode_index"].item()
        episode_length = self._dataset.meta.episodes[ep_idx]["length"]
        ep_start = self._dataset.episode_data_index["from"][ep_idx]
        frame_index_in_episode = idx - ep_start.item()

        # Check if we have keypoint rewards for this episode
        if ep_idx in self.keypoint_rewards and self.keypoint_rewards[ep_idx]:
            # Use cached interpolated rewards if available
            if ep_idx not in self._episode_reward_cache:
                keypoints = self.keypoint_rewards[ep_idx]
                self._episode_reward_cache[ep_idx] = self._interpolate_rewards(
                    keypoints, episode_length, "linear"
                )
            
            # Get reward for this frame
            if frame_index_in_episode < len(self._episode_reward_cache[ep_idx]):
                reward = self._episode_reward_cache[ep_idx][frame_index_in_episode]
            else:
                # Frame index out of bounds, use last reward
                reward = self._episode_reward_cache[ep_idx][-1]
        else:
            # Fallback to original linear interpolation based on progress
            progress = frame_index_in_episode / (episode_length - 1) if episode_length > 1 else 0.0

            if progress <= self.reward_start_pct:
                reward = 0.0
            elif progress >= self.reward_end_pct:
                reward = 1.0
            else:
                # Linear interpolation between start and end percentages
                interpolation_progress = (progress - self.reward_start_pct) / (
                    self.reward_end_pct - self.reward_start_pct
                )
                reward = interpolation_progress

        return torch.tensor(reward, dtype=torch.float32)

    def _get_reward_file_path(self) -> Path:
        """Get the path to the reward keypoints JSON file."""
        # Get the dataset root path
        dataset_root = Path(self._dataset.root)
        meta_dir = dataset_root / "meta"
        meta_dir.mkdir(exist_ok=True)
        return meta_dir / "reward_keypoints.json"

    def _load_keypoint_rewards(self) -> Dict[int, Dict[int, float]]:
        """
        Load keypoint rewards from JSON file.
        
        Returns:
            Dict mapping episode_index -> {frame_index: reward_value}
        """
        reward_file = self._get_reward_file_path()
        if reward_file.exists():
            try:
                with open(reward_file, 'r') as f:
                    data = json.load(f)
                # Convert string keys back to integers
                return {int(ep_idx): {int(frame_idx): float(reward) 
                                    for frame_idx, reward in keypoints.items()}
                       for ep_idx, keypoints in data.items()}
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Warning: Could not load reward keypoints file: {e}")
                return {}
        return {}

    def _save_keypoint_rewards(self):
        """Save keypoint rewards to JSON file."""
        reward_file = self._get_reward_file_path()
        reward_file.parent.mkdir(exist_ok=True)
        
        # Convert to string keys for JSON serialization
        data = {str(ep_idx): {str(frame_idx): float(reward) 
                             for frame_idx, reward in keypoints.items()}
               for ep_idx, keypoints in self.keypoint_rewards.items()}
        
        with open(reward_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_episode_rewards(self, episode_index: int, frame_rewards: Dict[int, float]):
        """
        Add or update reward keypoints for an episode.
        
        Args:
            episode_index: Index of the episode
            frame_rewards: Dict mapping frame_index (within episode) -> reward_value
        """
        # Validate reward values
        for frame_idx, reward in frame_rewards.items():
            if not (0.0 <= reward <= 1.0):
                raise ValueError(f"Reward value {reward} for frame {frame_idx} must be between 0.0 and 1.0")
        
        # Validate episode exists
        if episode_index >= len(self._dataset.meta.episodes):
            raise ValueError(f"Episode index {episode_index} does not exist in dataset")
        
        # Update keypoints
        if episode_index not in self.keypoint_rewards:
            self.keypoint_rewards[episode_index] = {}
        
        self.keypoint_rewards[episode_index].update(frame_rewards)
        
        # Clear cached rewards for this episode
        if episode_index in self._episode_reward_cache:
            del self._episode_reward_cache[episode_index]
        
        # Save to file
        self._save_keypoint_rewards()

    def remove_episode_rewards(self, episode_index: int):
        """Remove all reward keypoints for an episode."""
        if episode_index in self.keypoint_rewards:
            del self.keypoint_rewards[episode_index]
            if episode_index in self._episode_reward_cache:
                del self._episode_reward_cache[episode_index]
            self._save_keypoint_rewards()

    def get_episode_keypoints(self, episode_index: int) -> Dict[int, float]:
        """Get keypoint rewards for a specific episode."""
        return self.keypoint_rewards.get(episode_index, {})

    def _interpolate_rewards(self, keypoints: Dict[int, float], episode_length: int, 
                           interpolation_method: str = "linear") -> List[float]:
        """
        Interpolate sparse rewards to create dense rewards.
        
        Args:
            keypoints: Dict mapping frame_index -> reward_value
            episode_length: Total length of the episode
            interpolation_method: Method to use for interpolation ("linear" or "smooth")
            
        Returns:
            List of interpolated reward values for each frame
        """
        if not keypoints:
            return [0.0] * episode_length
        
        # Sort keypoints by frame index
        sorted_keypoints = sorted(keypoints.items())
        
        # Initialize rewards array
        rewards = np.zeros(episode_length)
        
        if len(sorted_keypoints) == 1:
            # Only one keypoint, fill all frames with that value
            frame_idx, reward_val = sorted_keypoints[0]
            rewards[:] = reward_val
            return rewards.tolist()
        
        # Handle interpolation between keyframes
        for i in range(len(sorted_keypoints) - 1):
            start_frame, start_reward = sorted_keypoints[i]
            end_frame, end_reward = sorted_keypoints[i + 1]
            
            # Check for sharp transitions (reward decreases by more than 0.5)
            # This typically indicates episode reset from success (1.0) to start (0.0)
            if start_reward > end_reward and (start_reward - end_reward) > 0.5:
                # Sharp transition: keep start_reward until start_frame, 
                # then jump to end_reward at end_frame
                for j in range(start_frame, min(end_frame, episode_length)):
                    rewards[j] = start_reward
                if end_frame < episode_length:
                    rewards[end_frame] = end_reward
            else:
                # Normal interpolation
                if interpolation_method == "linear":
                    # Linear interpolation
                    for j in range(start_frame, min(end_frame + 1, episode_length)):
                        if end_frame == start_frame:
                            rewards[j] = start_reward
                        else:
                            t = (j - start_frame) / (end_frame - start_frame)
                            rewards[j] = start_reward + t * (end_reward - start_reward)
                            
                elif interpolation_method == "smooth":
                    # Smooth interpolation using smoothstep function
                    for j in range(start_frame, min(end_frame + 1, episode_length)):
                        if end_frame == start_frame:
                            rewards[j] = start_reward
                        else:
                            t = (j - start_frame) / (end_frame - start_frame)
                            # smoothstep function for smoother transitions
                            smooth_t = 3 * t * t - 2 * t * t * t
                            rewards[j] = start_reward + smooth_t * (end_reward - start_reward)
                else:
                    raise ValueError(f"Unknown interpolation method: {interpolation_method}")
        
        # Fill frames before first keypoint with first reward value
        first_frame, first_reward = sorted_keypoints[0]
        rewards[:first_frame] = first_reward
        
        # Fill frames after last keypoint with last reward value
        last_frame, last_reward = sorted_keypoints[-1]
        rewards[last_frame:] = last_reward
        
        # Ensure rewards are within bounds
        rewards = np.clip(rewards, 0.0, 1.0)
        
        return rewards.tolist()

    def __len__(self) -> int:
        """Return the length of the underlying dataset."""
        return len(self._dataset)
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying dataset."""
        return getattr(self._dataset, name)
