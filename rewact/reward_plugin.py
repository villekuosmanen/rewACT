"""
Reward plugin for robocandywrapper.

This plugin adds reward calculation functionality to LeRobotDatasets with support
for user-defined keypoints and smooth interpolation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from scipy.interpolate import CubicSpline, interp1d
from robocandywrapper import DatasetPlugin, PluginInstance

from rewact.dataset_with_reward import KeypointReward


class RewardPluginInstance(PluginInstance):
    """
    Plugin instance that adds reward calculation to a dataset.
    
    Supports keypoint-based interpolation with smooth curves between keypoints.
    """
    
    def __init__(
        self, 
        dataset, 
        reward_start_pct: float = 0.05,
        reward_end_pct: float = 0.95,
        mask_actions_for_eval_data: bool = False,
        mask_actions_for_fail_data: bool = False,
    ):
        super().__init__(dataset)
        self.reward_start_pct = reward_start_pct
        self.reward_end_pct = reward_end_pct
        self.mask_actions_for_eval_data = mask_actions_for_eval_data
        self.mask_actions_for_fail_data = mask_actions_for_fail_data
        
        # Load keypoint rewards if they exist
        self.keypoint_rewards = self._load_keypoint_rewards()
        self._episode_reward_cache = {}
        
        # Create mapping from episode_idx to position in episode_data_index
        # This is needed when using a subset of episodes
        if hasattr(dataset, 'episodes') and dataset.episodes is not None:
            self._episode_idx_to_pos = {ep_idx: pos for pos, ep_idx in enumerate(dataset.episodes)}
        else:
            # No subsetting, episode_idx == position
            self._episode_idx_to_pos = None
    
    def get_data_keys(self) -> list[str]:
        """Return the keys this plugin will add to items."""
        return ["reward", "use_action_mask"]
    
    def _get_episode_data_index_pos(self, episode_idx: int) -> int:
        """
        Get the position in episode_data_index for a given episode_idx.
        
        When using a subset of episodes, episode_idx (the actual episode number)
        needs to be mapped to its position in the filtered episode_data_index.
        
        Args:
            episode_idx: The actual episode number from the dataset
            
        Returns:
            Position to use for indexing into episode_data_index
        """
        if self._episode_idx_to_pos is not None:
            if episode_idx not in self._episode_idx_to_pos:
                raise ValueError(
                    f"Episode {episode_idx} not found in the subset of episodes being used. "
                    f"Available episodes: {list(self._episode_idx_to_pos.keys())}"
                )
            return self._episode_idx_to_pos[episode_idx]
        else:
            # No subsetting, episode_idx is the position
            return episode_idx
    
    def get_item_data(self, idx: int, episode_idx: int) -> dict[str, Any]:
        """Get reward data for a specific item."""
        # Get episode information
        episode_length = self.dataset.meta.episodes[episode_idx]["length"]
        # Map episode_idx to position in episode_data_index (needed for episode subsets)
        ep_pos = self._get_episode_data_index_pos(episode_idx)
        ep_start = self.dataset.episode_data_index["from"][ep_pos]
        
        # Calculate frame index within episode
        # Note: idx is local to the dataset, not global
        frame_index_in_episode = idx - ep_start.item()
        
        # Calculate reward
        reward = self._calculate_reward(episode_idx, frame_index_in_episode, episode_length)
        
        # Calculate action masking
        repo_name = self.dataset.repo_id.split('/')[1] if '/' in self.dataset.repo_id else self.dataset.repo_id
        use_action_mask = True
        
        if self.mask_actions_for_eval_data and repo_name.startswith('eval_'):
            use_action_mask = False
        elif self.mask_actions_for_fail_data and repo_name.startswith('fail_'):
            use_action_mask = False
        
        return {
            "reward": torch.tensor(reward, dtype=torch.float32),
            "use_action_mask": torch.tensor(use_action_mask, dtype=torch.bool)
        }
    
    def _calculate_reward(self, episode_idx: int, frame_index: int, episode_length: int) -> float:
        """
        Calculate reward based on keypoint interpolation or fallback linear interpolation.
        
        Args:
            episode_idx: Episode index
            frame_index: Frame index within the episode
            episode_length: Total length of the episode
            
        Returns:
            Reward value (float)
        """
        # Check if we have keypoint rewards for this episode
        if episode_idx in self.keypoint_rewards and self.keypoint_rewards[episode_idx]:
            # Use cached interpolated rewards if available
            if episode_idx not in self._episode_reward_cache:
                keypoints = self.keypoint_rewards[episode_idx]
                self._episode_reward_cache[episode_idx] = self._interpolate_rewards(
                    keypoints, episode_length, "smooth"
                )
            
            # Get reward for this frame
            if frame_index < len(self._episode_reward_cache[episode_idx]):
                reward = self._episode_reward_cache[episode_idx][frame_index]
            else:
                # Frame index out of bounds, use last reward
                reward = self._episode_reward_cache[episode_idx][-1]
        else:
            # Fallback to linear interpolation based on progress
            progress = frame_index / (episode_length - 1) if episode_length > 1 else 0.0

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

        return reward
    
    def _get_reward_file_path(self) -> Path:
        """Get the path to the reward keypoints JSON file."""
        dataset_root = Path(self.dataset.root)
        rewact_extensions_dir = dataset_root / "rewact_extensions"
        rewact_extensions_dir.mkdir(exist_ok=True)
        return rewact_extensions_dir / "reward_keypoints.json"
    
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
                
                result = {}
                for ep_idx, ep_data in data.items():
                    episode_index = int(ep_idx)
                    result[episode_index] = {}
                    
                    if "keypoints" not in ep_data or not isinstance(ep_data["keypoints"], list):
                        raise ValueError(f"Invalid format for episode {ep_idx}: missing 'keypoints' list")
                    
                    # Load keypoint format: {"keypoints": [{"frame_index": 10, "reward": 0.5}, ...]}
                    for kp_data in ep_data["keypoints"]:
                        frame_idx = kp_data["frame_index"]
                        reward = kp_data["reward"]
                        result[episode_index][frame_idx] = float(reward)
                    
                return result
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Warning: Could not load reward keypoints file: {e}")
                return {}
        return {}
    
    def _save_keypoint_rewards(self):
        """Save keypoint rewards to JSON file."""
        reward_file = self._get_reward_file_path()
        reward_file.parent.mkdir(exist_ok=True)
        
        # Convert to keypoint format for JSON serialization
        data = {}
        for ep_idx, keypoints in self.keypoint_rewards.items():
            keypoint_list = []
            for frame_idx, reward in keypoints.items():
                # Convert frame index back to timestamp for storage
                timestamp = self._frame_index_to_timestamp(frame_idx, ep_idx)
                keypoint_list.append({
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "reward": float(reward)
                })
            
            data[str(ep_idx)] = {
                "keypoints": keypoint_list,
                "fps": self.dataset.fps
            }
        
        with open(reward_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _timestamp_to_frame_index(self, timestamp: float, episode_index: int) -> int:
        """Convert timestamp to frame index within an episode."""
        fps = self.dataset.fps
        frame_index = int(round(timestamp * fps))
        
        # Validate against episode length
        episode_length = self.dataset.meta.episodes[episode_index]["length"]
        if frame_index >= episode_length:
            raise ValueError(
                f"Timestamp {timestamp}s corresponds to frame {frame_index}, "
                f"but episode {episode_index} only has {episode_length} frames"
            )
        
        return frame_index
    
    def _frame_index_to_timestamp(self, frame_index: int, episode_index: int) -> float:
        """Convert frame index to timestamp within an episode."""
        fps = self.dataset.fps
        return frame_index / fps
    
    def _normalize_keypoints(self, keypoints: List[KeypointReward], episode_index: int) -> Dict[int, float]:
        """
        Convert keypoint objects to normalized frame_index -> reward mapping.
        
        Args:
            keypoints: List of KeypointReward objects
            episode_index: Episode index for validation
            
        Returns:
            Dict mapping frame_index -> reward_value
        """
        normalized = {}
        
        for keypoint in keypoints:
            if keypoint.frame_index is not None:
                frame_idx = keypoint.frame_index
            else:  # timestamp is not None
                frame_idx = self._timestamp_to_frame_index(keypoint.timestamp, episode_index)
            
            # Check for duplicate frame indices
            if frame_idx in normalized:
                raise ValueError(f"Multiple keypoints specified for frame {frame_idx}")
            
            normalized[frame_idx] = keypoint.reward
        
        return normalized
    
    def add_episode_rewards(self, episode_index: int, keypoints):
        """
        Add or update reward keypoints for an episode.
        
        Args:
            episode_index: Index of the episode
            keypoints: Either a list of KeypointReward objects or a dict mapping frame_index -> reward_value
        """
        # Validate episode exists
        if episode_index >= len(self.dataset.meta.episodes):
            raise ValueError(f"Episode index {episode_index} does not exist in dataset")
        
        # Handle both new and legacy formats
        if isinstance(keypoints, dict):
            frame_rewards = keypoints
            for frame_idx, reward in frame_rewards.items():
                if not (0.0 <= reward <= 1.0):
                    raise ValueError(f"Reward value {reward} for frame {frame_idx} must be between 0.0 and 1.0")
        else:
            frame_rewards = self._normalize_keypoints(keypoints, episode_index)
        
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
        """Get keypoint rewards for a specific episode in frame_index -> reward format."""
        return self.keypoint_rewards.get(episode_index, {})
    
    def get_episode_keypoints_objects(self, episode_index: int) -> List[KeypointReward]:
        """
        Get keypoint rewards for a specific episode as KeypointReward objects.
        
        Returns:
            List of KeypointReward objects with both frame_index and timestamp information
        """
        frame_rewards = self.keypoint_rewards.get(episode_index, {})
        keypoints = []
        
        for frame_idx, reward in frame_rewards.items():
            timestamp = self._frame_index_to_timestamp(frame_idx, episode_index)
            keypoints.append(KeypointReward(
                reward=reward,
                frame_index=frame_idx,
                timestamp=timestamp
            ))
        
        return sorted(keypoints, key=lambda kp: kp.frame_index)
    
    # Interpolation methods - copied from dataset_with_reward.py
    
    def _find_sharp_transitions(self, sorted_keypoints: List[tuple], threshold: float = 0.99) -> List[int]:
        """Find indices where sharp transitions occur."""
        sharp_transitions = []
        for i in range(len(sorted_keypoints) - 1):
            _, y0 = sorted_keypoints[i]
            _, y1 = sorted_keypoints[i + 1]
            if y0 > y1 and (y0 - y1) > threshold:
                sharp_transitions.append(i + 1)
        return sharp_transitions
    
    def _add_boundary_keypoints(self, sorted_keypoints: List[tuple], episode_length: int) -> List[tuple]:
        """Add boundary keypoints at frame 0 and episode end."""
        if not sorted_keypoints:
            return [(0, 0.0), (episode_length - 1, 0.0)]
        
        enhanced_keypoints = list(sorted_keypoints)
        
        if sorted_keypoints[0][0] != 0:
            enhanced_keypoints.insert(0, (0, 0.0))
        
        last_frame = episode_length - 1
        if sorted_keypoints[-1][0] != last_frame:
            last_reward = sorted_keypoints[-1][1]
            enhanced_keypoints.append((last_frame, last_reward))
        
        return enhanced_keypoints
    
    def _add_derivative_control_points(self, segment: List[tuple]) -> List[tuple]:
        """Add control points to ensure zero derivatives at segment boundaries."""
        if len(segment) < 2:
            return segment
        
        enhanced_segment = list(segment)
        
        if len(segment) >= 3:
            start_frame, start_reward = segment[0]
            second_frame, _ = segment[1]
            
            control_frame_start = start_frame + max(1, (second_frame - start_frame) // 3)
            if control_frame_start != start_frame and control_frame_start != second_frame:
                enhanced_segment.insert(1, (control_frame_start, start_reward))
            
            end_frame, end_reward = segment[-1]
            second_last_frame, _ = segment[-2]
            
            control_frame_end = end_frame - max(1, (end_frame - second_last_frame) // 3)
            if control_frame_end != end_frame and control_frame_end != second_last_frame:
                enhanced_segment.insert(-1, (control_frame_end, end_reward))
        
        return enhanced_segment
    
    def _split_keypoints_at_transitions(self, sorted_keypoints: List[tuple], episode_length: int) -> List[List[tuple]]:
        """Split keypoints into segments at sharp transitions."""
        enhanced_keypoints = self._add_boundary_keypoints(sorted_keypoints, episode_length)
        sharp_transitions = self._find_sharp_transitions(enhanced_keypoints)
        
        segments = []
        start_idx = 0
        
        if not sharp_transitions:
            segment = self._add_derivative_control_points(enhanced_keypoints)
            segments.append(segment)
        else:
            for transition_idx in sharp_transitions:
                if transition_idx > start_idx:
                    segment = enhanced_keypoints[start_idx:transition_idx]
                    segment = self._add_derivative_control_points(segment)
                    segments.append(segment)
                start_idx = transition_idx
            
            if start_idx < len(enhanced_keypoints):
                segment = enhanced_keypoints[start_idx:]
                segment = self._add_derivative_control_points(segment)
                segments.append(segment)
        
        return [seg for seg in segments if len(seg) >= 2]
    
    def _interpolate_rewards(self, keypoints: Dict[int, float], episode_length: int, 
                           interpolation_method: str = "smooth") -> List[float]:
        """Interpolate sparse rewards to create dense rewards."""
        if not keypoints:
            return [0.0] * episode_length
        
        sorted_keypoints = sorted(keypoints.items())
        rewards = np.zeros(episode_length)
        
        if len(sorted_keypoints) == 1:
            frame_idx, reward_val = sorted_keypoints[0]
            rewards[:] = reward_val
            return rewards.tolist()
        
        if interpolation_method == "smooth":
            segments = self._split_keypoints_at_transitions(sorted_keypoints, episode_length)
            
            for segment in segments:
                if len(segment) < 2:
                    continue
                    
                x_segment = np.array([kp[0] for kp in segment])
                y_segment = np.array([kp[1] for kp in segment])
                
                if len(segment) == 2:
                    interp_func = interp1d(x_segment, y_segment, kind='linear', 
                                         bounds_error=False, fill_value='extrapolate')
                else:
                    try:
                        interp_func = CubicSpline(x_segment, y_segment, bc_type='natural')
                    except Exception:
                        interp_func = interp1d(x_segment, y_segment, kind='linear',
                                             bounds_error=False, fill_value='extrapolate')
                
                start_frame = int(x_segment[0])
                end_frame = int(x_segment[-1])
                frame_indices = np.arange(start_frame, min(end_frame + 1, episode_length))
                
                if len(frame_indices) > 0:
                    interpolated_values = interp_func(frame_indices)
                    for i, frame_idx in enumerate(frame_indices):
                        if 0 <= frame_idx < episode_length:
                            rewards[frame_idx] = interpolated_values[i]
            
            # Handle sharp transitions between segments
            for i in range(len(segments) - 1):
                current_segment = segments[i]
                next_segment = segments[i + 1]
                
                if len(current_segment) > 0 and len(next_segment) > 0:
                    transition_frame = int(next_segment[0][0])
                    if 0 <= transition_frame < episode_length:
                        rewards[transition_frame] = next_segment[0][1]
        else:
            # Linear interpolation
            for i in range(len(sorted_keypoints) - 1):
                start_frame, start_reward = sorted_keypoints[i]
                end_frame, end_reward = sorted_keypoints[i + 1]
                
                if start_reward > end_reward and (start_reward - end_reward) > 0.5:
                    for j in range(start_frame, min(end_frame, episode_length)):
                        rewards[j] = start_reward
                    if end_frame < episode_length:
                        rewards[end_frame] = end_reward
                else:
                    for j in range(start_frame, min(end_frame + 1, episode_length)):
                        if end_frame == start_frame:
                            rewards[j] = start_reward
                        else:
                            t = (j - start_frame) / (end_frame - start_frame)
                            rewards[j] = start_reward + t * (end_reward - start_reward)
        
        rewards = np.clip(rewards, 0.0, 1.0)
        return rewards.tolist()


class RewardPlugin(DatasetPlugin):
    """
    Plugin that adds reward calculation to LeRobotDatasets.
    
    Supports keypoint-based interpolation with smooth curves.
    
    Example:
        ```python
        from robocandywrapper import WrappedRobotDataset
        from rewact.reward_plugin import RewardPlugin
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        # Create plugin
        reward_plugin = RewardPlugin(reward_start_pct=0.05, reward_end_pct=0.95)
        
        # Load dataset with plugin
        base_dataset = LeRobotDataset("my_dataset")
        dataset = WrappedRobotDataset(base_dataset, plugins=[reward_plugin])
        
        # Access data with rewards
        item = dataset[0]
        reward = item["reward"]  # Tensor with interpolated reward value
        ```
    """
    
    def __init__(
        self, 
        reward_start_pct: float = 0.05, 
        reward_end_pct: float = 0.95,
        mask_actions_for_eval_data: bool = False,
        mask_actions_for_fail_data: bool = False,
    ):
        """
        Initialize the reward plugin.
        
        Args:
            reward_start_pct: Progress percentage where reward starts (0.0) - used for fallback
            reward_end_pct: Progress percentage where reward reaches maximum (1.0) - used for fallback
            mask_actions_for_eval_data: If True, mask actions from datasets with 'eval_' prefix
            mask_actions_for_fail_data: If True, mask actions from datasets with 'fail_' prefix
        """
        if not (0.0 <= reward_start_pct <= 1.0 and 0.0 <= reward_end_pct <= 1.0):
            raise ValueError("Reward percentages must be between 0.0 and 1.0")
        if reward_start_pct >= reward_end_pct:
            raise ValueError("reward_start_pct must be less than reward_end_pct")
        
        self.reward_start_pct = reward_start_pct
        self.reward_end_pct = reward_end_pct
        self.mask_actions_for_eval_data = mask_actions_for_eval_data
        self.mask_actions_for_fail_data = mask_actions_for_fail_data
    
    def attach(self, dataset) -> RewardPluginInstance:
        """Create a dataset-specific plugin instance."""
        return RewardPluginInstance(
            dataset,
            reward_start_pct=self.reward_start_pct,
            reward_end_pct=self.reward_end_pct,
            mask_actions_for_eval_data=self.mask_actions_for_eval_data,
            mask_actions_for_fail_data=self.mask_actions_for_fail_data,
        )

