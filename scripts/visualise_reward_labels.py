#!/usr/bin/env python

"""
Script to visualize labeled rewards from keypoints on dataset episodes.
Loads a LeRobotDataset with reward labels and creates visualization videos
showing the interpolated rewards between keypoints.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rewact.dataset_with_reward import LeRobotDatasetWithReward

# Import visualization functions from reward_wrapper
from reward_wrapper import create_reward_visualization_video


def extract_images_from_frame(frame: dict) -> List[torch.Tensor]:
    """Extract image tensors from a dataset frame."""
    images = []
    for key, value in frame.items():
        if "image" in key and isinstance(value, torch.Tensor):
            # Convert to proper format for visualization
            img = value.clone()
            
            # Handle different tensor formats
            if img.dim() == 4:  # (B,C,H,W)
                img = img.squeeze(0)
            
            # Ensure proper channel format (C,H,W)
            if img.dim() == 3:
                if img.shape[2] in [1, 3]:  # (H,W,C) format
                    img = img.permute(2, 0, 1)  # Convert to (C,H,W)
                # else already in (C,H,W) format
                
                # Normalize to [0,1] range if needed
                if img.max() > 1.0:
                    img = img.float() / 255.0
                
                images.append(img)
    
    return images


def analyze_episode_rewards(
    reward_dataset: LeRobotDatasetWithReward,
    episode_id: int,
    output_dir: str = "outputs"
) -> Dict:
    """
    Analyze reward labels for an episode and create visualization.
    
    Args:
        reward_dataset: LeRobotDatasetWithReward instance
        episode_id: Episode ID to analyze
        output_dir: Directory to save output videos
        
    Returns:
        Dictionary containing analysis results
    """
    
    # Get episode information
    episode_length = reward_dataset._dataset.meta.episodes[episode_id]["length"]
    episode_start_idx = reward_dataset._dataset.episode_data_index["from"][episode_id].item()
    
    print(f"Analyzing episode {episode_id} with {episode_length} frames")
    
    # Check if episode has keypoint rewards
    keypoints = reward_dataset.get_episode_keypoints(episode_id)
    
    if keypoints:
        print(f"Episode has {len(keypoints)} reward keypoints:")
        for frame_idx, reward in sorted(keypoints.items()):
            print(f"  Frame {frame_idx}: {reward:.3f}")
    else:
        print("Episode has no keypoint rewards - using fallback linear interpolation")
    
    # Collect reward data and images
    reward_data = []
    reward_images = []
    
    print("Processing frames...")
    for frame_idx in tqdm(range(episode_length)):
        global_idx = episode_start_idx + frame_idx
        
        # Get frame with reward
        frame = reward_dataset[global_idx]
        reward_value = frame["reward"].item()
        
        # Store reward data
        reward_data.append({
            'step': frame_idx,
            'reward': reward_value
        })
        
        # Extract images for visualization
        images = extract_images_from_frame(frame)
        reward_images.append(images)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualization video
    output_filename = os.path.join(output_dir, f"episode_{episode_id}_reward_labels.mp4")
    
    print(f"Creating reward visualization video...")
    create_reward_visualization_video(
        reward_images, 
        reward_data, 
        output_filename, 
        fps=20
    )
    
    # Calculate statistics
    rewards = [r['reward'] for r in reward_data]
    
    stats = {
        'episode_id': episode_id,
        'episode_length': episode_length,
        'num_keypoints': len(keypoints),
        'keypoints': keypoints,
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
        'final_reward': rewards[-1] if rewards else 0.0,
        'video_path': output_filename
    }
    
    # Print statistics
    print(f"\nReward Statistics for Episode {episode_id}:")
    print(f"  Mean: {stats['mean_reward']:.3f}")
    print(f"  Std: {stats['std_reward']:.3f}")
    print(f"  Min: {stats['min_reward']:.3f}")
    print(f"  Max: {stats['max_reward']:.3f}")
    print(f"  Final: {stats['final_reward']:.3f}")
    print(f"  Video saved to: {output_filename}")
    
    return stats


def visualize_keypoint_progression(
    reward_dataset: LeRobotDatasetWithReward,
    episode_id: int,
    output_dir: str = "outputs"
):
    """
    Create a detailed analysis of how rewards progress through keypoints.
    """
    keypoints = reward_dataset.get_episode_keypoints(episode_id)
    episode_length = reward_dataset._dataset.meta.episodes[episode_id]["length"]
    
    if not keypoints:
        print(f"Episode {episode_id} has no keypoints to analyze")
        return
    
    print(f"\nDetailed Keypoint Analysis for Episode {episode_id}:")
    print("=" * 60)
    
    # Sort keypoints by frame index
    sorted_keypoints = sorted(keypoints.items())
    
    # Analyze transitions between keypoints
    for i, (frame_idx, reward) in enumerate(sorted_keypoints):
        progress_pct = (frame_idx / (episode_length - 1)) * 100 if episode_length > 1 else 0
        
        print(f"Keypoint {i+1}: Frame {frame_idx} ({progress_pct:.1f}%) -> Reward {reward:.3f}")
        
        if i > 0:
            prev_frame, prev_reward = sorted_keypoints[i-1]
            frame_diff = frame_idx - prev_frame
            reward_diff = reward - prev_reward
            
            if abs(reward_diff) > 0.5:
                transition_type = "SHARP TRANSITION" if reward_diff < 0 else "LARGE INCREASE"
                print(f"  └─ {transition_type}: {reward_diff:+.3f} over {frame_diff} frames")
            else:
                print(f"  └─ Smooth transition: {reward_diff:+.3f} over {frame_diff} frames")
    
    # Show interpolated reward curve
    print(f"\nInterpolated reward progression (every 10 frames):")
    episode_start_idx = reward_dataset._dataset.episode_data_index["from"][episode_id].item()
    
    for frame_idx in range(0, episode_length, 10):
        global_idx = episode_start_idx + frame_idx
        frame = reward_dataset[global_idx]
        reward_value = frame["reward"].item()
        
        # Check if this frame is a keypoint
        is_keypoint = frame_idx in keypoints
        marker = " ★" if is_keypoint else ""
        
        print(f"  Frame {frame_idx:3d}: {reward_value:.3f}{marker}")


def add_example_keypoints(reward_dataset: LeRobotDatasetWithReward, episode_id: int):
    """
    Add example keypoints to an episode for demonstration purposes.
    """
    episode_length = reward_dataset._dataset.meta.episodes[episode_id]["length"]
    
    print(f"Adding example keypoints to episode {episode_id} (length: {episode_length})...")
    
    # Create a reasonable progression: start low, build up, succeed, then reset
    example_keypoints = {
        0: 0.0,                                    # Start
        episode_length // 4: 0.2,                 # Early progress
        episode_length // 2: 0.5,                 # Halfway
        int(episode_length * 0.75): 0.8,          # Near completion
        episode_length - 10: 1.0,                 # Success
        episode_length - 9: 0.0,                  # Reset (demonstrates sharp transition)
        episode_length - 1: 0.1                   # Small progress on retry
    }
    
    # Only add keypoints that are within the episode length
    valid_keypoints = {k: v for k, v in example_keypoints.items() if k < episode_length}
    
    reward_dataset.add_episode_rewards(episode_id, valid_keypoints)
    
    print(f"Added {len(valid_keypoints)} keypoints:")
    for frame_idx, reward in sorted(valid_keypoints.items()):
        print(f"  Frame {frame_idx}: {reward:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize labeled rewards from keypoints")
    parser.add_argument("--dataset-repo-id", type=str, required=True,
                        help="Repository ID of the dataset to analyze")
    parser.add_argument("--episode-id", type=int, default=None,
                        help="Episode ID to analyze (if not specified, analyzes first episode)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save output videos")
    parser.add_argument("--add-example-keypoints", action="store_true",
                        help="Add example keypoints to the episode for demonstration")
    parser.add_argument("--reward-start-pct", type=float, default=0.05,
                        help="Fallback reward start percentage")
    parser.add_argument("--reward-end-pct", type=float, default=0.95,
                        help="Fallback reward end percentage")
    parser.add_argument("--analyze-all-episodes", action="store_true",
                        help="Analyze all episodes in the dataset")
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset_repo_id}")
    
    # Load dataset
    base_dataset = LeRobotDataset(args.dataset_repo_id)
    print(f"Dataset loaded successfully. Total episodes: {base_dataset.num_episodes}")
    
    # Wrap with reward functionality
    reward_dataset = LeRobotDatasetWithReward(
        base_dataset,
        reward_start_pct=args.reward_start_pct,
        reward_end_pct=args.reward_end_pct
    )
    
    # Determine which episodes to analyze
    if args.analyze_all_episodes:
        episodes_to_analyze = list(range(base_dataset.num_episodes))
        print(f"Will analyze all {base_dataset.num_episodes} episodes")
    else:
        if args.episode_id is not None:
            if args.episode_id >= base_dataset.num_episodes:
                raise ValueError(f"Episode {args.episode_id} not found. Dataset has {base_dataset.num_episodes} episodes.")
            episode_id = args.episode_id
        else:
            episode_id = 0  # Default to first episode
            
        episodes_to_analyze = [episode_id]
        print(f"Will analyze episode {episode_id}")
    
    # Process each episode
    all_stats = []
    
    for episode_id in episodes_to_analyze:
        print(f"\n{'='*80}")
        print(f"Processing Episode {episode_id}")
        print(f"{'='*80}")
        
        # Add example keypoints if requested
        if args.add_example_keypoints:
            add_example_keypoints(reward_dataset, episode_id)
        
        # Analyze the episode
        stats = analyze_episode_rewards(
            reward_dataset, 
            episode_id, 
            args.output_dir
        )
        all_stats.append(stats)
        
        # Show detailed keypoint analysis
        visualize_keypoint_progression(reward_dataset, episode_id, args.output_dir)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for stats in all_stats:
        ep_id = stats['episode_id']
        num_keypoints = stats['num_keypoints']
        mean_reward = stats['mean_reward']
        video_path = stats['video_path']
        
        keypoint_status = f"{num_keypoints} keypoints" if num_keypoints > 0 else "fallback interpolation"
        print(f"Episode {ep_id}: {keypoint_status}, mean reward: {mean_reward:.3f}")
        print(f"  Video: {video_path}")
    
    print(f"\nAll videos saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
