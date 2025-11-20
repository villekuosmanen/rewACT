#!/usr/bin/env python

"""
Migration script to extract reward keypoints from dense rewards in parquet files
and move them to the rewact_extensions JSON format.

This script:
1. Loads a LeRobotDataset with dense rewards in parquet files
2. Extracts keypoints from the dense rewards (episode by episode)
3. Saves keypoints to rewact_extensions/reward_keypoints.json
4. Removes the reward column from parquet files
5. Does NOT push to hub (manual step)

Usage:
    python scripts/migrate_rewards_to_keypoints.py --dataset-repo-id <repo_id> [--dry-run]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robocandywrapper import WrappedRobotDataset
from rewact.plugins import DenseRewardPlugin, get_plugin_instance

# Tolerance for exact keypoint matching (very tight for run detection)
EPSILON = 5e-6
# Tolerance for quarter-value detection (more lenient to catch floating point variations)
QUARTER_VALUE_TOLERANCE = 0.01

def normalize_reward_value(reward: float, offset_rewards: bool = False) -> float:
    """
    Normalize reward value from offset range (0.01, 0.26, 0.51, 0.76, 1.0)
    back to standard range (0, 0.25, 0.5, 0.75, 1.0).
    
    Args:
        reward: Original reward value
        offset_rewards: Whether rewards use the offset format
        
    Returns:
        Normalized reward value
    """
    if not offset_rewards:
        return reward
    
    # Map offset rewards back to standard range
    # 0.01 -> 0.0, 0.26 -> 0.25, 0.51 -> 0.5, 0.76 -> 0.75, 1.0 -> 1.0
    if abs(reward - 0.01) < EPSILON:
        return 0.0
    elif abs(reward - 0.26) < EPSILON:
        return 0.25
    elif abs(reward - 0.51) < EPSILON:
        return 0.5
    elif abs(reward - 0.76) < EPSILON:
        return 0.75
    elif abs(reward - 1.0) < EPSILON:
        return 1.0
    else:
        # For interpolated values, apply linear transformation
        return (reward - 0.01) / 0.99


def detect_reward_format(rewards: np.ndarray) -> bool:
    """
    Detect if rewards use offset format (0.01, 0.26, 0.51, 0.76, 1.0)
    or standard format (0, 0.25, 0.5, 0.75, 1.0).
    
    Args:
        rewards: Array of reward values
        
    Returns:
        True if offset format, False if standard format
    """
    unique_rewards = np.unique(rewards)
    
    # Check for offset format markers
    offset_markers = [0.01, 0.26, 0.51, 0.76]
    offset_count = sum(1 for marker in offset_markers if any(abs(unique_rewards - marker) < EPSILON))
    
    # Check for standard format markers
    standard_markers = [0.0, 0.25, 0.5, 0.75]
    standard_count = sum(1 for marker in standard_markers if any(abs(unique_rewards - marker) < EPSILON))
    
    # Decide based on which format has more matches
    return offset_count > standard_count


def extract_keypoints_from_rewards(rewards: np.ndarray, tolerance: float = EPSILON) -> Dict[int, float]:
    """
    Extract keypoints from dense reward array.
    
    Strategy:
    - Keypoints were originally set in 1/4th increments (0, 0.25, 0.5, 0.75, 1.0)
    - Episodes can contain BOTH standard format (0, 0.25, 0.5, 0.75, 1.0) AND
      offset format (0.01, 0.26, 0.51, 0.76, 1.0) values mixed together
    - When the same reward value appears many times in a row, both the first and last
      instance qualify as keypoints
    - This helps preserve the shape of the reward curve
    
    Args:
        rewards: Array of reward values for an episode
        tolerance: Tolerance for considering two values as equal
        
    Returns:
        Dict mapping frame_index -> reward_value for detected keypoints
    """
    if len(rewards) == 0:
        return {}
    
    keypoints = {}
    
    # Helper function to normalize any reward value
    def normalize_any_reward(reward_val: float) -> float:
        """Normalize a reward value whether it's in standard or offset format."""
        # Check if it matches offset format
        if abs(reward_val - 0.01) < EPSILON:
            return 0.0
        elif abs(reward_val - 0.26) < EPSILON:
            return 0.25
        elif abs(reward_val - 0.51) < EPSILON:
            return 0.5
        elif abs(reward_val - 0.76) < EPSILON:
            return 0.75
        elif abs(reward_val - 1.0) < EPSILON:
            return 1.0
        # Check if it matches standard format
        elif abs(reward_val - 0.0) < EPSILON:
            return 0.0
        elif abs(reward_val - 0.25) < EPSILON:
            return 0.25
        elif abs(reward_val - 0.5) < EPSILON:
            return 0.5
        elif abs(reward_val - 0.75) < EPSILON:
            return 0.75
    
    # Always include first and last frame
    keypoints[0] = normalize_any_reward(rewards[0])
    keypoints[len(rewards) - 1] = normalize_any_reward(rewards[-1])
    
    # Find runs of constant reward values
    i = 0
    while i < len(rewards):
        current_reward = rewards[i]
        run_start = i
        
        # Find the end of this run
        while i < len(rewards) and abs(rewards[i] - current_reward) < tolerance:
            i += 1
        run_end = i - 1
        
        # If this is a significant run (more than 1 frame), mark both start and end
        if run_end > run_start:
            normalized_reward = normalize_any_reward(current_reward)
            
            # Add start of run (if not already added)
            if run_start not in keypoints:
                keypoints[run_start] = normalized_reward
            
            # Add end of run (if not already added and not last frame)
            if run_end not in keypoints and run_end < len(rewards) - 1:
                keypoints[run_end] = normalized_reward
    
    # Look for BOTH standard and offset quarter-point reward values
    # Standard format values
    standard_quarter_values = [
        (0.0, 0.0),
        (0.25, 0.25),
        (0.5, 0.5),
        (0.75, 0.75),
        (1.0, 1.0),
    ]
    
    # Offset format values (normalized to standard)
    offset_quarter_values = [
        (0.01, 0.0),
        (0.26, 0.25),
        (0.51, 0.5),
        (0.76, 0.75),
        (1.0, 1.0),
    ]
    
    # Check for both formats in the same episode
    all_quarter_values = standard_quarter_values + offset_quarter_values
    
    # Find first and last occurrence of each quarter value (both formats)
    # Use a more lenient tolerance to catch floating point variations
    # IMPORTANT: We detect start/end of each CONTINUOUS SEGMENT of quarter values,
    # not just first/last in entire episode, to handle multiple cycles
    for raw_val, norm_val in all_quarter_values:
        matches = np.where(np.abs(rewards - raw_val) < EPSILON)[0]
        if len(matches) == 0:
            continue
        
        # Find continuous segments of this quarter value
        # A segment ends when there's a gap larger than 1 frame
        segments = []
        current_segment = [matches[0]]
        
        for i in range(1, len(matches)):
            if matches[i] - matches[i-1] <= 1:
                # Continue current segment
                current_segment.append(matches[i])
            else:
                # Gap found, save current segment and start new one
                segments.append(current_segment)
                current_segment = [matches[i]]
        # Don't forget the last segment
        segments.append(current_segment)
        
        # Add start and end of each segment as keypoints
        for segment in segments:
            # Start of segment
            start_idx = int(segment[0])
            if start_idx not in keypoints:
                keypoints[start_idx] = norm_val
            
            # End of segment (if more than one frame in segment)
            if len(segment) > 1:
                end_idx = int(segment[-1])
                if end_idx not in keypoints:
                    keypoints[end_idx] = norm_val
    
    # Additional pass: detect local maxima and minima to catch peaks
    # This helps capture reward cycles that might not hit exact quarter values
    if len(rewards) > 2:
        for i in range(1, len(rewards) - 1):
            # Local maximum (peak)
            if rewards[i] > rewards[i-1] and rewards[i] > rewards[i+1]:
                if i not in keypoints:
                    keypoints[i] = normalize_any_reward(rewards[i])
            # Local minimum (trough)
            elif rewards[i] < rewards[i-1] and rewards[i] < rewards[i+1]:
                if i not in keypoints:
                    keypoints[i] = normalize_any_reward(rewards[i])
    
    return keypoints


def extract_episode_rewards(dataset: LeRobotDataset, episode_idx: int) -> np.ndarray:
    """
    Extract reward values for a specific episode.
    
    Args:
        dataset: LeRobotDataset instance
        episode_idx: Episode index
        
    Returns:
        Array of reward values for the episode
    """
    ep_start = dataset.episode_data_index["from"][episode_idx].item()
    ep_end = dataset.episode_data_index["to"][episode_idx].item()
    
    # Get rewards for this episode
    rewards = []
    for idx in range(ep_start, ep_end):
        item = dataset.hf_dataset[idx]
        if "reward" in item:
            reward_val = item["reward"]
            if isinstance(reward_val, torch.Tensor):
                reward_val = reward_val.item()
            rewards.append(reward_val)
        else:
            raise ValueError(f"No 'reward' column found in dataset at index {idx}")
    
    return np.array(rewards)


def migrate_dataset_rewards(
    dataset_repo_id: str,
    dry_run: bool = False,
    verbose: bool = True
) -> Dict[int, Dict[int, float]]:
    """
    Migrate rewards from parquet files to rewact_extensions JSON format.
    
    Args:
        dataset_repo_id: Repository ID of the dataset
        dry_run: If True, only analyze without modifying files
        verbose: If True, print detailed progress
        
    Returns:
        Dict mapping episode_index -> {frame_index: reward_value}
    """
    if verbose:
        print("=" * 80)
        print(f"Migrating rewards for dataset: {dataset_repo_id}")
        print("=" * 80)
    
    # Load dataset
    if verbose:
        print("\n1. Loading dataset...")
    base_dataset = LeRobotDataset(dataset_repo_id)
    
    if verbose:
        print(f"   ✓ Dataset loaded")
        print(f"   Total episodes: {base_dataset.num_episodes}")
        print(f"   Total frames: {base_dataset.num_frames}")
        print(f"   FPS: {base_dataset.fps}")
    
    # Check if reward column exists
    if "reward" not in base_dataset.hf_dataset.column_names:
        print(f"\n⚠ Warning: No 'reward' column found in dataset. Nothing to migrate.")
        return {}
    
    # Extract keypoints from all episodes
    all_keypoints = {}
    
    if verbose:
        print(f"\n2. Extracting keypoints from {base_dataset.num_episodes} episodes...")
    
    for episode_idx in range(base_dataset.num_episodes):
        # Extract rewards for this episode
        rewards = extract_episode_rewards(base_dataset, episode_idx)
        
        # Extract keypoints
        keypoints = extract_keypoints_from_rewards(rewards)
        
        if len(keypoints) > 0:
            all_keypoints[episode_idx] = keypoints
            
            if verbose:
                episode_length = len(rewards)
                episode_duration = episode_length / base_dataset.fps
                print(f"   Episode {episode_idx:3d}: {len(keypoints):2d} keypoints "
                      f"({episode_length} frames, {episode_duration:.1f}s)")
                if episode_idx < 3:  # Show details for first 3 episodes
                    for frame_idx, reward in sorted(keypoints.items())[:15]:
                        timestamp = frame_idx / base_dataset.fps
                        print(f"      Frame {frame_idx:4d} ({timestamp:5.1f}s): {reward:.3f}")
                    # if len(keypoints) > 5:
                    #     print(f"      ... and {len(keypoints) - 5} more")
    
    total_keypoints = sum(len(kps) for kps in all_keypoints.values())
    if verbose:
        print(f"\n   ✓ Extracted {total_keypoints} total keypoints from {len(all_keypoints)} episodes")
    
    if dry_run:
        print("\n🏃 DRY RUN MODE - No files will be modified")
        return all_keypoints
    
    # Save keypoints using the plugin system
    if verbose:
        print("\n3. Saving keypoints to rewact_extensions/reward_keypoints.json...")
    
    # Wrap dataset with plugin to save keypoints
    reward_plugin_obj = DenseRewardPlugin()
    wrapped_dataset = WrappedRobotDataset(base_dataset, plugins=[reward_plugin_obj])
    reward_plugin = get_plugin_instance(wrapped_dataset, DenseRewardPlugin, dataset_idx=0)
    
    # Add keypoints for each episode
    for episode_idx, keypoints in all_keypoints.items():
        reward_plugin.add_episode_rewards(episode_idx, keypoints)
    
    reward_file = reward_plugin._get_reward_file_path()
    if verbose:
        print(f"   ✓ Keypoints saved to: {reward_file}")
    
    # Remove reward column from parquet files
    if verbose:
        print("\n4. Removing 'reward' column from parquet files...")
    
    # Remove reward column from the hf_dataset
    base_dataset.hf_dataset = base_dataset.hf_dataset.remove_columns(["reward"])
    
    # Re-save each episode's parquet file without the reward column
    for episode_idx in range(base_dataset.num_episodes):
        ep_start = base_dataset.episode_data_index["from"][episode_idx].item()
        ep_end = base_dataset.episode_data_index["to"][episode_idx].item()
        
        # Get episode data without reward
        ep_dataset = base_dataset.hf_dataset.select(range(ep_start, ep_end))
        
        # Save to parquet
        ep_data_path = base_dataset.root / base_dataset.meta.get_data_file_path(ep_index=episode_idx)
        ep_dataset.to_parquet(ep_data_path)
        
        if verbose and episode_idx % 10 == 0:
            print(f"   Processed episode {episode_idx}/{base_dataset.num_episodes}")
    
    if verbose:
        print(f"   ✓ Removed 'reward' column from all parquet files")
    
    # Update info.json to remove reward from features
    if verbose:
        print("\n5. Updating metadata...")
    
    if "reward" in base_dataset.meta.info["features"]:
        del base_dataset.meta.info["features"]["reward"]
        from lerobot.datasets.utils import write_info
        write_info(base_dataset.meta.info, base_dataset.root)
        if verbose:
            print("   ✓ Updated info.json to remove reward feature")
    
    return all_keypoints


def main():
    parser = argparse.ArgumentParser(
        description="Migrate dense rewards from parquet to keypoints in JSON"
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="Repository ID of the dataset to migrate"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and show what would be done without modifying files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress (default: True)"
    )
    
    args = parser.parse_args()
    
    try:
        all_keypoints = migrate_dataset_rewards(
            args.dataset_repo_id,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        
        if args.verbose:
            print("\n" + "=" * 80)
            print("MIGRATION SUMMARY")
            print("=" * 80)
            print(f"Episodes with keypoints: {len(all_keypoints)}")
            print(f"Total keypoints extracted: {sum(len(kps) for kps in all_keypoints.values())}")
            
            if not args.dry_run:
                print("\n✓ Migration completed successfully!")
                print("\nNext steps:")
                print("  1. Review the migrated keypoints using:")
                print(f"     python scripts/visualise_reward_labels.py --dataset-repo-id {args.dataset_repo_id}")
                print("  2. Test training with the new format:")
                print(f"     python scripts/train.py dataset_repo_id={args.dataset_repo_id}")
                print("  3. If everything looks good, push to hub:")
                print("     (Manual step - use HuggingFace hub CLI or API)")
            else:
                print("\n✓ Dry run completed - no files were modified")
                print("  Remove --dry-run flag to perform actual migration")
        
        return 0
    
    except Exception as e:
        print(f"\n✗ Error during migration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

