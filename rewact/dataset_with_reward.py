from pathlib import Path
from typing import Callable

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotDatasetWithReward:
    """
    A wrapper around LeRobotDataset that adds reward calculation based on linear interpolation.
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
            reward_start_pct: Progress percentage where reward starts (0.0)
            reward_end_pct: Progress percentage where reward reaches maximum (1.0)
            All other args are passed through to LeRobotDataset
        """
        # Create the underlying dataset
        self._dataset = dataset

        # Validate reward parameters
        if not (0.0 <= reward_start_pct <= 1.0 and 0.0 <= reward_end_pct <= 1.0):
            raise ValueError("Reward percentages must be between 0.0 and 1.0")
        if reward_start_pct >= reward_end_pct:
            raise ValueError("reward_start_pct must be less than reward_end_pct")

        # Store reward parameters
        self.reward_start_pct = reward_start_pct
        self.reward_end_pct = reward_end_pct


    def __getitem__(self, idx) -> dict:
        """Get an item from the dataset with calculated reward."""
        item = self._dataset[idx]
        
        # Add reward calculation
        item["reward"] = self._calculate_reward(idx, item)
        
        return item
    
    def _calculate_reward(self, idx: int, item: dict) -> torch.Tensor:
        """
        Calculate reward based on linear interpolation within episode.
        
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

        # Calculate progress as a fraction of the episode (0.0 to 1.0)
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

    def __len__(self) -> int:
        """Return the length of the underlying dataset."""
        return len(self._dataset)
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying dataset."""
        return getattr(self._dataset, name)
