#!/usr/bin/env python
"""
Pre-compute advantages for all frames in a dataset using a trained value function.

Saves advantages per episode to a directory to avoid memory issues.

Usage:
    # Run once for each dataset you want to train on
    python precompute_advantage.py --dataset lerobot/dataset1 --value_model checkpoints/value_function --output outputs/dataset1_advantages
    python precompute_advantage.py --dataset lerobot/dataset2 --value_model checkpoints/value_function --output outputs/dataset2_advantages
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

from robocandywrapper.dataformats.lerobot_21 import LeRobot21Dataset
from robocandywrapper import WrappedRobotDataset
from robocandywrapper.plugins import EpisodeOutcomePlugin

from rewact.plugins import PiStar0_6CumulativeRewardPlugin
from rewact.policy import RewACTPolicy  # Your value function model


def smooth_values(values: list[float], window_size: int) -> list[float]:
    """
    Apply temporal smoothing to value predictions using a moving average.
    
    Args:
        values: List of value predictions for an episode
        window_size: Size of smoothing window (1 = no smoothing, 3 = ±1 frame, 5 = ±2 frames, etc.)
    
    Returns:
        Smoothed values (same length as input)
    """
    if window_size <= 1:
        return values
    
    # Convert to numpy for efficient computation
    values_arr = np.array(values)
    smoothed = np.zeros_like(values_arr)
    
    # Half window size (how many frames before/after to include)
    half_window = window_size // 2
    
    for i in range(len(values)):
        # Determine the window bounds (handle edges)
        start_idx = max(0, i - half_window)
        end_idx = min(len(values), i + half_window + 1)
        
        # Average over the window
        smoothed[i] = values_arr[start_idx:end_idx].mean()
    
    return smoothed.tolist()


def compute_advantages(
    dataset_path: str,
    value_model_path: str,
    output_dir: str,
    n_step_lookahead: int = 50,
    value_smoothing_window: int = 1,
    device: str = "cuda",
):
    """
    Compute advantages for all frames in dataset.
    
    Advantage = [N-step return + V(s_{t+N})] - V(s_t)
    
    Saves advantages per episode to avoid memory issues.
    
    Args:
        dataset_path: Path to the dataset
        value_model_path: Path to trained value function model
        output_dir: Directory to save advantage files
        n_step_lookahead: Number of steps ahead for N-step return
        value_smoothing_window: Window size for temporal smoothing of value predictions.
                               1 = no smoothing, 3 = average current + 1 before + 1 after,
                               5 = average current + 2 before + 2 after, etc.
        device: Device to run model on
    """
    print(f"Loading dataset: {dataset_path}")
    dataset = LeRobot21Dataset(dataset_path)
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
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track statistics across all episodes
    all_advantages = []
    
    if value_smoothing_window > 1:
        print(f"Using value smoothing with window size {value_smoothing_window} (±{value_smoothing_window//2} frames)")
    
    print("Computing advantages...")
    for episode_idx in tqdm(range(wrapped_dataset.num_episodes)):
        if episode_idx < 1:
            continue
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
        
        # Apply temporal smoothing to values if requested
        if value_smoothing_window > 1:
            episode_values = smooth_values(episode_values, value_smoothing_window)
        
        # Compute advantages for each frame in this episode
        episode_advantages = []
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
            
            episode_advantages.append({
                'frame_index': episode_start + t,
                'episode_idx': episode_idx,
                'advantage': advantage,
            })
            all_advantages.append(advantage)
        
        # Save this episode's advantages immediately
        episode_df = pd.DataFrame(episode_advantages)
        episode_file = output_path / f"episode_{episode_idx:05d}.parquet"
        episode_df.to_parquet(episode_file, index=False)
        
        # for stability
        import time
        time.sleep(2)
    
    # Print overall statistics
    print(f"\nAdvantage statistics:")
    print(f"  Mean: {sum(all_advantages)/len(all_advantages):.4f}")
    print(f"  Min: {min(all_advantages):.4f}")
    print(f"  Max: {max(all_advantages):.4f}")
    print(f"  Positive: {sum(1 for a in all_advantages if a > 0)/len(all_advantages)*100:.1f}%")
    
    print(f"\nSaved {wrapped_dataset.num_episodes} episode advantage files to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset repo_id")
    parser.add_argument("--value_model", type=str, required=True, help="Path to trained value function")
    parser.add_argument("--output", type=str, required=True, help="Output directory for episode advantage files")
    parser.add_argument("--n_step", type=int, default=50, help="N-step lookahead for advantage")
    parser.add_argument("--value_smoothing", type=int, default=3, 
                       help="Temporal smoothing window for value predictions (1=no smoothing, 3=±1 frame, 5=±2 frames)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    compute_advantages(
        dataset_path=args.dataset,
        value_model_path=args.value_model,
        output_dir=args.output,
        n_step_lookahead=args.n_step,
        value_smoothing_window=args.value_smoothing,
        device=args.device,
    )
