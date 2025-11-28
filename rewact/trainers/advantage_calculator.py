#!/usr/bin/env python
"""Advantage Calculator for pre-computing advantages from a trained value function."""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robocandywrapper import WrappedRobotDataset
from robocandywrapper.plugins import EpisodeOutcomePlugin

from rewact.plugins import PiStar0_6CumulativeRewardPlugin
from rewact.policy import RewACTPolicy


class AdvantageCalculator:
    """Calculator for pre-computing advantages using a trained value function."""

    def __init__(
        self,
        dataset_path: str,
        value_model_path: str,
        output_dir: str,
        n_step_lookahead: int = 50,
        device: str = "cuda",
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        """
        Initialize advantage calculator.

        Args:
            dataset_path: Path to dataset (HuggingFace repo_id or local path)
            value_model_path: Path to trained value function model
            output_dir: Directory to save advantage parquet files
            n_step_lookahead: N-step lookahead for advantage computation
            device: Device to use for computation
            batch_size: Batch size for batched GPU inference (default: 16)
            num_workers: Number of dataloader workers (default: 4)
        """
        self.dataset_path = dataset_path
        self.value_model_path = value_model_path
        self.output_dir = Path(output_dir)
        self.n_step_lookahead = n_step_lookahead
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

    def compute(self) -> Dict[str, any]:
        """
        Compute advantages for all frames in the dataset using batched GPU inference.

        Advantage = [N-step return + V(s_{t+N})] - V(s_t)

        Returns:
            Dictionary with computation results including output_path and statistics
        """
        logging.info(f"Loading dataset: {self.dataset_path}")
        dataset = LeRobotDataset(self.dataset_path)
        wrapped_dataset = WrappedRobotDataset(
            datasets=[dataset],
            plugins=[EpisodeOutcomePlugin(), PiStar0_6CumulativeRewardPlugin(normalise=False)],
        )

        # For denormalisation
        denormalise_plugin = PiStar0_6CumulativeRewardPlugin(normalise=True).attach(dataset)
        denormalise_plugin._compute_normalization_parameters()

        logging.info(f"Loading value function model: {self.value_model_path}")
        value_model = RewACTPolicy.from_pretrained(self.value_model_path)
        value_model = value_model.to(self.device)
        value_model.eval()

        # Create dataloader for batched inference (no shuffling to maintain order)
        dataloader = torch.utils.data.DataLoader(
            wrapped_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,  # Important: maintain sequential order
            pin_memory=self.device == "cuda",
            drop_last=False,
        )

        num_episodes = wrapped_dataset.num_episodes
        logging.info(f"Computing values for {len(wrapped_dataset)} frames across {num_episodes} episodes...")
        logging.info(f"Using batch size {self.batch_size} with {self.num_workers} dataloader workers")
        
        # Collect all values and rewards in order
        all_values = []
        all_rewards = []
        
        # Process batches through the model
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                # Move batch to device
                batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                               for k, v in batch.items()}
                
                # Get value predictions for the batch
                _, reward_output = value_model.predict_action_chunk(batch_device)
                
                # Extract expected values and denormalize
                values = reward_output['expected_value'].squeeze().cpu()
                if values.dim() == 0:  # Single item batch
                    values = values.unsqueeze(0)
                
                for value in values:
                    denorm_value = denormalise_plugin.denormalize_reward(value.item())
                    all_values.append(denorm_value)
                
                # Get rewards from batch
                rewards = batch['reward'].squeeze().cpu()
                if rewards.dim() == 0:  # Single item batch
                    rewards = rewards.unsqueeze(0)
                all_rewards.extend(rewards.tolist())
        
        logging.info(f"Computed {len(all_values)} value predictions")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Now compute advantages per episode
        advantages = []
        for episode_idx in tqdm(range(num_episodes), desc="Computing advantages per episode"):
            episode_length = wrapped_dataset._datasets[0].meta.episodes[episode_idx]["length"]
            episode_start = wrapped_dataset._datasets[0].episode_data_index["from"][episode_idx].item()
            
            # Get values and rewards for this episode
            episode_values = all_values[episode_start:episode_start + episode_length]
            episode_rewards = all_rewards[episode_start:episode_start + episode_length]
            
            # Compute advantages for each frame in episode
            episode_advantages = []
            for t in range(episode_length):
                # N-step return
                n_step_end = min(t + self.n_step_lookahead, episode_length - 1)
                n_step_return = (episode_rewards[n_step_end] - episode_rewards[t]) * -1
                
                # Add value of future state (if not at end)
                if n_step_end < episode_length - 1:
                    future_value = episode_values[n_step_end]
                else:
                    future_value = 0.0  # Terminal state
                
                # Advantage = [N-step return + V(future)] - V(current)
                advantage = (n_step_return + future_value) - episode_values[t]
                
                episode_advantages.append({
                    'frame_index': episode_start + t,
                    'episode_idx': episode_idx,
                    'advantage': advantage,
                })
            
            # Write episode to parquet file
            episode_df = pd.DataFrame(episode_advantages)
            episode_file = self.output_dir / f"episode_{episode_idx:05d}.parquet"
            episode_df.to_parquet(episode_file, index=False)
            
            advantages.extend(episode_advantages)
        
        # Create DataFrame from all advantages for statistics
        df = pd.DataFrame(advantages)

        # Statistics
        stats = {
            "mean": df['advantage'].mean(),
            "std": df['advantage'].std(),
            "min": df['advantage'].min(),
            "max": df['advantage'].max(),
            "positive_pct": (df['advantage'] > 0).mean() * 100,
            "num_frames": len(df),
        }

        logging.info(f"\nAdvantage statistics:")
        logging.info(f"  Mean: {stats['mean']:.4f}")
        logging.info(f"  Std: {stats['std']:.4f}")
        logging.info(f"  Min: {stats['min']:.4f}")
        logging.info(f"  Max: {stats['max']:.4f}")
        logging.info(f"  Positive: {stats['positive_pct']:.1f}%")
        
        logging.info(f"\nSaved {len(df['episode_idx'].unique())} episode files to: {self.output_dir}")

        return {
            "output_path": str(self.output_dir),
            "statistics": stats,
            "dataset_path": self.dataset_path,
            "value_model_path": self.value_model_path,
        }

    def push_to_hub(self, repo_id: str, commit_message: Optional[str] = None) -> str:
        """
        Push computed advantages to HuggingFace Hub as a dataset.

        Args:
            repo_id: HuggingFace repo ID (e.g., "username/dataset-name-advantages")
            commit_message: Optional commit message

        Returns:
            URL to the created repository
        """
        from huggingface_hub import HfApi, create_repo
        
        if not self.output_dir.exists():
            raise ValueError(f"Output directory {self.output_dir} does not exist. Run compute() first.")
        
        logging.info(f"Pushing advantages to HuggingFace Hub: {repo_id}")
        
        # Create repo if it doesn't exist
        try:
            create_repo(repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
            logging.warning(f"Could not create repo (may already exist): {e}")
        
        # Upload all parquet files
        api = HfApi()
        
        parquet_files = list(self.output_dir.glob("*.parquet"))
        logging.info(f"Uploading {len(parquet_files)} parquet files...")
        
        for parquet_file in tqdm(parquet_files):
            api.upload_file(
                path_or_fileobj=str(parquet_file),
                path_in_repo=parquet_file.name,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message or f"Add {parquet_file.name}",
            )
        
        # Create a README with metadata
        readme_content = f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- advantage
- reinforcement-learning
- rewact
---

# Advantage Values for {self.dataset_path}

Pre-computed advantage values for offline RL training.

## Source
- **Dataset**: {self.dataset_path}
- **Value Model**: {self.value_model_path}
- **N-step lookahead**: {self.n_step_lookahead}

## Files
This dataset contains per-episode parquet files with advantage values for each frame.

## Usage
```python
from pathlib import Path
import pandas as pd

# Load advantages for a specific episode
advantage_df = pd.read_parquet("episode_00000.parquet")
```
"""
        
        readme_path = self.output_dir / "README.md"
        readme_path.write_text(readme_content)
        
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message or "Add README",
        )
        
        hub_url = f"https://huggingface.co/datasets/{repo_id}"
        logging.info(f"Advantages pushed to: {hub_url}")
        
        return hub_url

