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
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
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
        self._dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
        )

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
    
    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        reward_start_pct: float = 0.05,
        reward_end_pct: float = 0.95,
    ) -> "LeRobotDatasetWithReward":
        """
        Create a LeRobot Dataset from scratch for recording data with reward calculation.
        
        This method creates the underlying LeRobotDataset and wraps it with reward functionality.
        """
        # Create the underlying dataset using LeRobotDataset.create
        base_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type=robot_type,
            use_videos=use_videos,
            tolerance_s=tolerance_s,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
        )
        
        # Create the wrapper instance
        obj = cls.__new__(cls)
        obj._dataset = base_dataset
        obj.reward_start_pct = reward_start_pct
        obj.reward_end_pct = reward_end_pct
        
        # Validate reward parameters
        if not (0.0 <= reward_start_pct <= 1.0 and 0.0 <= reward_end_pct <= 1.0):
            raise ValueError("Reward percentages must be between 0.0 and 1.0")
        if reward_start_pct >= reward_end_pct:
            raise ValueError("reward_start_pct must be less than reward_end_pct")
        
        return obj
