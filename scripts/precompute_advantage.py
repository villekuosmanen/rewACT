#!/usr/bin/env python
"""
Pre-compute advantages for all frames in a dataset using a trained value function.

Usage:
    python compute_advantages.py --dataset lerobot/my_dataset --value_model checkpoints/value_function --output advantages.parquet
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robocandywrapper import WrappedRobotDataset
from robocandywrapper.plugins import EpisodeOutcomePlugin

from rewact.plugins import PiStar0_6CumulativeRewardPlugin
from rewact.policy import RewACTPolicy  # Your value function model


def compute_advantages(
    dataset_path: str,
    value_model_path: str,
    output_file: str,
    n_step_lookahead: int = 50,
    device: str = "cuda",
):
    """
    Compute advantages for all frames in dataset.
    
    Advantage = [N-step return + V(s_{t+N})] - V(s_t)
    """
    print(f"Loading dataset: {dataset_path}")
    dataset = LeRobotDataset(dataset_path)
    wrapped_dataset = WrappedRobotDataset(
        datasets=[dataset],
        plugins=[EpisodeOutcomePlugin(), PiStar0_6CumulativeRewardPlugin(normalise=False)],
    )

    # for denormalisation
    denormalise_plugin = PiStar0_6CumulativeRewardPlugin(normalise=True).attach(dataset)
    denormalise_plugin._compute_normalization_parameters()

    print(f"Loading value function model: {value_model_path}")
    value_model = RewACTPolicy.from_pretrained(value_model_path)
    value_model = value_model.to(device)
    value_model.eval()
    
    advantages = []
    
    print("Computing advantages...")
    for episode_idx in tqdm(range(wrapped_dataset.num_episodes)):
        episode_length = wrapped_dataset._datasets[0].meta.episodes[episode_idx]["length"]
        episode_start = wrapped_dataset._datasets[0].episode_data_index["from"][episode_idx].item()
        
        # Get all frames for this episode
        episode_indices = range(episode_start, episode_start + episode_length)
        
        # Compute values for all frames in episode
        episode_values = []
        episode_rewards = []
        
        for idx in episode_indices:
            # Get frame data
            item = wrapped_dataset[idx]
            
            # Prepare batch (add batch dimension)
            batch = {k: v.unsqueeze(0).to(device) for k, v in item.items() if isinstance(v, torch.Tensor)}
            
            # Get value prediction
            with torch.no_grad():
                # Assuming your value function returns reward_output with 'expected_value'
                _, reward_output = value_model.select_action(batch)
                value_model.reset()
                value = reward_output.cpu().item()
                value = denormalise_plugin.denormalize_reward(value)
            
            episode_values.append(value)
            
            # Get reward (cumulative reward from your plugin)
            episode_rewards.append(item['reward'].item())
        
        # Compute advantages for each frame
        for t in range(episode_length):
            # N-step return
            n_step_end = min(t + n_step_lookahead, episode_length - 1)
            n_step_return = (episode_rewards[n_step_end] - episode_rewards[t]) * -1
            
            # Add value of future state (if not at end)
            if n_step_end < episode_length - 1:
                future_value = episode_values[n_step_end]
            else:
                future_value = 0.0  # Terminal state
            
            # Advantage = [N-step return + V(future)] - V(current)
            advantage = (n_step_return + future_value) - episode_values[t]
            
            advantages.append({
                'frame_index': episode_start + t,
                'episode_idx': episode_idx,
                'advantage': advantage,
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(advantages)
    
    print(f"\nAdvantage statistics:")
    print(f"  Mean: {df['advantage'].mean():.4f}")
    print(f"  Std: {df['advantage'].std():.4f}")
    print(f"  Min: {df['advantage'].min():.4f}")
    print(f"  Max: {df['advantage'].max():.4f}")
    print(f"  Positive: {(df['advantage'] > 0).mean()*100:.1f}%")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved advantages to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset repo_id")
    parser.add_argument("--value_model", type=str, required=True, help="Path to trained value function")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file")
    parser.add_argument("--n_step", type=int, default=50, help="N-step lookahead for advantage")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    compute_advantages(
        dataset_path=args.dataset,
        value_model_path=args.value_model,
        output_file=args.output,
        n_step_lookahead=args.n_step,
        device=args.device,
    )
