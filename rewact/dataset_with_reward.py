import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from scipy.interpolate import CubicSpline, interp1d


@dataclass
class KeypointReward:
    """
    A keypoint reward specification that can be defined by either frame index or timestamp.
    
    Args:
        reward: Reward value between 0.0 and 1.0
        frame_index: Frame index within the episode (0-based), mutually exclusive with timestamp
        timestamp: Timestamp in seconds from episode start, mutually exclusive with frame_index
    """
    reward: float
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        # Validate reward value
        if not (0.0 <= self.reward <= 1.0):
            raise ValueError(f"Reward value {self.reward} must be between 0.0 and 1.0")
        
        # Validate that exactly one of frame_index or timestamp is set
        has_frame_index = self.frame_index is not None
        has_timestamp = self.timestamp is not None
        
        if not (has_frame_index ^ has_timestamp):  # XOR: exactly one should be True
            raise ValueError("Exactly one of 'frame_index' or 'timestamp' must be specified")
        
        # Validate non-negative values
        if has_frame_index and self.frame_index < 0:
            raise ValueError(f"Frame index {self.frame_index} must be non-negative")
        
        if has_timestamp and self.timestamp < 0.0:
            raise ValueError(f"Timestamp {self.timestamp} must be non-negative")


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
    - Automatic saving/loading of keypoint rewards to/from JSON files in the dataset's rewact_extensions directory
    - Smooth interpolation between keypoints with support for sharp transitions
    - Caching of interpolated rewards for performance
    - Validation of reward values (must be between 0.0 and 1.0)
    - Non-intrusive: works alongside existing LeRobot datasets without modification
    
    Example usage:
        ```python
        # Load dataset and wrap with reward functionality
        dataset = LeRobotDataset("my_dataset")
        reward_dataset = LeRobotDatasetWithReward(dataset)
        
        # Method 1: Add keypoint rewards using KeypointReward objects (preferred)
        from rewact.dataset_with_reward import KeypointReward
        
        keypoints = [
            KeypointReward(reward=0.0, frame_index=5),      # Start at frame 5
            KeypointReward(reward=0.33, timestamp=4.35),    # Milestone at 4.35 seconds
            KeypointReward(reward=0.66, frame_index=104),   # Second milestone  
            KeypointReward(reward=1.0, timestamp=8.0)       # Success at 8 seconds
        ]
        reward_dataset.add_episode_rewards(3, keypoints)
        
        # Method 2: Legacy format (frame_index -> reward dict)
        episode_rewards = {5: 0.0, 87: 0.33, 104: 0.66, 160: 1.0}
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
                    keypoints, episode_length, "smooth"
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

    def _timestamp_to_frame_index(self, timestamp: float, episode_index: int) -> int:
        """
        Convert timestamp to frame index within an episode.
        
        Args:
            timestamp: Timestamp in seconds from episode start
            episode_index: Episode index
            
        Returns:
            Frame index within the episode
        """
        fps = self._dataset.fps
        frame_index = int(round(timestamp * fps))
        
        # Validate against episode length
        episode_length = self._dataset.meta.episodes[episode_index]["length"]
        if frame_index >= episode_length:
            raise ValueError(
                f"Timestamp {timestamp}s corresponds to frame {frame_index}, "
                f"but episode {episode_index} only has {episode_length} frames"
            )
        
        return frame_index

    def _frame_index_to_timestamp(self, frame_index: int, episode_index: int) -> float:
        """
        Convert frame index to timestamp within an episode.
        
        Args:
            frame_index: Frame index within the episode
            episode_index: Episode index
            
        Returns:
            Timestamp in seconds from episode start
        """
        fps = self._dataset.fps
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

    def _get_reward_file_path(self) -> Path:
        """Get the path to the reward keypoints JSON file."""
        # Get the dataset root path
        dataset_root = Path(self._dataset.root)
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
        """Save keypoint rewards to JSON file in the new keypoint format."""
        reward_file = self._get_reward_file_path()
        reward_file.parent.mkdir(exist_ok=True)
        
        # Convert to new keypoint format for JSON serialization
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
                "fps": self._dataset.fps  # Store FPS for reference
            }
        
        with open(reward_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_episode_rewards(self, episode_index: int, keypoints: Union[List[KeypointReward], Dict[int, float]]):
        """
        Add or update reward keypoints for an episode.
        
        Args:
            episode_index: Index of the episode
            keypoints: Either a list of KeypointReward objects or a dict mapping frame_index -> reward_value (legacy format)
        """
        # Validate episode exists
        if episode_index >= len(self._dataset.meta.episodes):
            raise ValueError(f"Episode index {episode_index} does not exist in dataset")
        
        # Handle both new and legacy formats
        if isinstance(keypoints, dict):
            # Legacy format: Dict[int, float]
            frame_rewards = keypoints
            # Validate reward values (KeypointReward validation happens in __post_init__)
            for frame_idx, reward in frame_rewards.items():
                if not (0.0 <= reward <= 1.0):
                    raise ValueError(f"Reward value {reward} for frame {frame_idx} must be between 0.0 and 1.0")
        else:
            # New format: List[KeypointReward]
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

    def _find_sharp_transitions(self, sorted_keypoints: List[Tuple[int, float]], threshold: float = 0.99) -> List[int]:
        """
        Find indices where sharp transitions occur (reward drops by more than threshold).
        
        Args:
            sorted_keypoints: List of (frame_index, reward) tuples sorted by frame_index
            threshold: Threshold for detecting sharp transitions
            
        Returns:
            List of indices where sharp transitions occur
        """
        sharp_transitions = []
        for i in range(len(sorted_keypoints) - 1):
            _, y0 = sorted_keypoints[i]
            _, y1 = sorted_keypoints[i + 1]
            if y0 > y1 and (y0 - y1) > threshold:
                sharp_transitions.append(i + 1)  # Index of the point after the drop
        return sharp_transitions

    def _add_boundary_keypoints(self, sorted_keypoints: List[Tuple[int, float]], episode_length: int) -> List[Tuple[int, float]]:
        """
        Add boundary keypoints at frame 0 and episode end to ensure smooth interpolation.
        
        Args:
            sorted_keypoints: List of (frame_index, reward) tuples sorted by frame_index
            episode_length: Total length of the episode
            
        Returns:
            List with boundary keypoints added
        """
        if not sorted_keypoints:
            return [(0, 0.0), (episode_length - 1, 0.0)]
        
        enhanced_keypoints = list(sorted_keypoints)
        
        # Add frame 0 if not present
        if sorted_keypoints[0][0] != 0:
            enhanced_keypoints.insert(0, (0, 0.0))
        
        # Add last frame if not present
        last_frame = episode_length - 1
        if sorted_keypoints[-1][0] != last_frame:
            # Use the last labeled reward value
            last_reward = sorted_keypoints[-1][1]
            enhanced_keypoints.append((last_frame, last_reward))
        
        return enhanced_keypoints

    def _add_derivative_control_points(self, segment: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Add control points to ensure zero derivatives at segment boundaries.
        This helps create smooth curves without plateaus.
        
        Args:
            segment: A segment of keypoints
            
        Returns:
            Segment with additional control points for derivative control
        """
        if len(segment) < 2:
            return segment
        
        enhanced_segment = list(segment)
        
        # For segments with 3+ points, add control points near boundaries
        if len(segment) >= 3:
            # Add a point near the start to control the derivative
            start_frame, start_reward = segment[0]
            second_frame, _ = segment[1]
            
            # Add a control point 1/3 of the way to the second point
            control_frame_start = start_frame + max(1, (second_frame - start_frame) // 3)
            if control_frame_start != start_frame and control_frame_start != second_frame:
                enhanced_segment.insert(1, (control_frame_start, start_reward))
            
            # Add a point near the end to control the derivative
            end_frame, end_reward = segment[-1]
            second_last_frame, _ = segment[-2]
            
            # Add a control point 1/3 of the way back from the end point
            control_frame_end = end_frame - max(1, (end_frame - second_last_frame) // 3)
            if control_frame_end != end_frame and control_frame_end != second_last_frame:
                enhanced_segment.insert(-1, (control_frame_end, end_reward))
        
        return enhanced_segment

    def _split_keypoints_at_transitions(self, sorted_keypoints: List[Tuple[int, float]], episode_length: int) -> List[List[Tuple[int, float]]]:
        """
        Split keypoints into segments at sharp transitions and add boundary/control points.
        
        Args:
            sorted_keypoints: List of (frame_index, reward) tuples sorted by frame_index
            episode_length: Total length of the episode
            
        Returns:
            List of keypoint segments with proper boundary and control points
        """
        # First add boundary keypoints
        enhanced_keypoints = self._add_boundary_keypoints(sorted_keypoints, episode_length)
        
        # Find sharp transitions in the enhanced keypoints
        sharp_transitions = self._find_sharp_transitions(enhanced_keypoints)
        
        segments = []
        start_idx = 0
        
        if not sharp_transitions:
            # No sharp transitions, create single segment
            segment = self._add_derivative_control_points(enhanced_keypoints)
            segments.append(segment)
        else:
            # Split at sharp transitions
            for transition_idx in sharp_transitions:
                # Include the point before the transition in the current segment
                if transition_idx > start_idx:
                    segment = enhanced_keypoints[start_idx:transition_idx]
                    segment = self._add_derivative_control_points(segment)
                    segments.append(segment)
                # Start new segment from the transition point
                start_idx = transition_idx
            
            # Add the final segment
            if start_idx < len(enhanced_keypoints):
                segment = enhanced_keypoints[start_idx:]
                segment = self._add_derivative_control_points(segment)
                segments.append(segment)
        
        return [seg for seg in segments if len(seg) >= 2]  # Only keep segments with 2+ points

    def _interpolate_rewards(self, keypoints: Dict[int, float], episode_length: int, 
                           interpolation_method: str = "smooth") -> List[float]:
        """
        Interpolate sparse rewards to create dense rewards using scipy for smooth interpolation.
        
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
        
        if interpolation_method == "smooth":
            # Use scipy for professional-quality smooth interpolation
            # First, split keypoints at sharp transitions
            segments = self._split_keypoints_at_transitions(sorted_keypoints, episode_length)
            
            for segment in segments:
                if len(segment) < 2:
                    continue
                    
                # Extract x (frames) and y (rewards) for this segment
                x_segment = np.array([kp[0] for kp in segment])
                y_segment = np.array([kp[1] for kp in segment])
                
                if len(segment) == 2:
                    # For two points, use linear interpolation
                    interp_func = interp1d(x_segment, y_segment, kind='linear', 
                                         bounds_error=False, fill_value='extrapolate')
                else:
                    # For 3+ points, use cubic spline with natural boundary conditions
                    # bc_type='natural' gives zero second derivatives at endpoints
                    # This creates the smoothest possible curve
                    try:
                        interp_func = CubicSpline(x_segment, y_segment, bc_type='natural')
                    except Exception:
                        # Fallback to linear if cubic spline fails
                        interp_func = interp1d(x_segment, y_segment, kind='linear',
                                             bounds_error=False, fill_value='extrapolate')
                
                # Interpolate for the range covered by this segment
                start_frame = int(x_segment[0])
                end_frame = int(x_segment[-1])
                
                # Generate all frame indices in this segment
                frame_indices = np.arange(start_frame, min(end_frame + 1, episode_length))
                
                if len(frame_indices) > 0:
                    interpolated_values = interp_func(frame_indices)
                    
                    # Assign interpolated values to rewards array
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
                        # Set the exact value at the transition point (sharp jump)
                        rewards[transition_frame] = next_segment[0][1]
        
        else:
            # Linear interpolation
            for i in range(len(sorted_keypoints) - 1):
                start_frame, start_reward = sorted_keypoints[i]
                end_frame, end_reward = sorted_keypoints[i + 1]
                
                # Check for sharp transitions
                if start_reward > end_reward and (start_reward - end_reward) > 0.5:
                    # Sharp transition
                    for j in range(start_frame, min(end_frame, episode_length)):
                        rewards[j] = start_reward
                    if end_frame < episode_length:
                        rewards[end_frame] = end_reward
                else:
                    # Linear interpolation
                    for j in range(start_frame, min(end_frame + 1, episode_length)):
                        if end_frame == start_frame:
                            rewards[j] = start_reward
                        else:
                            t = (j - start_frame) / (end_frame - start_frame)
                            rewards[j] = start_reward + t * (end_reward - start_reward)
        
        # Note: Boundary handling is now done in _split_keypoints_at_transitions
        # which adds boundary keypoints at frame 0 and episode end
        
        # Ensure rewards are within bounds
        rewards = np.clip(rewards, 0.0, 1.0)
        
        return rewards.tolist()


    def __len__(self) -> int:
        """Return the length of the underlying dataset."""
        return len(self._dataset)
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying dataset."""
        return getattr(self._dataset, name)
