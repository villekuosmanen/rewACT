#!/usr/bin/env python

"""
Script to visualize pre-computed advantage values from episodes.
Shows advantages over time with color-coded regions (green=positive, red=negative).

The visualization displays:
- Camera images at the top
- Advantage graph below with:
  - Yellow threshold line at the middle (percentile-based or fixed)
  - Green background for positive advantages (above threshold)
  - Red background for negative advantages (below threshold)
  - Advantage values clipped to [threshold-1, threshold+1] range
  - Line graph showing advantage trajectory (green=positive, red=negative)

Usage:
    # Using percentile threshold (default 30th percentile)
    python visualise_advantages.py \\
        --dataset-repo-id lerobot/dataset1 \\
        --advantage-dir outputs/dataset1_advantages \\
        --episode-id 5
    
    # Using custom percentile (e.g., 40th percentile = top 60% positive)
    python visualise_advantages.py \\
        --dataset-repo-id lerobot/dataset1 \\
        --advantage-dir outputs/dataset1_advantages \\
        --episode-id 5 \\
        --percentile 40.0
    
    # Using fixed threshold
    python visualise_advantages.py \\
        --dataset-repo-id lerobot/dataset1 \\
        --advantage-dir outputs/dataset1_advantages \\
        --episode-id 5 \\
        --threshold 0.5
"""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from robocandywrapper.dataformats.lerobot_21 import LeRobot21Dataset
from robocandywrapper import WrappedRobotDataset
from robocandywrapper.plugins import EpisodeOutcomePlugin

from rewact.plugins import PiStar0_6CumulativeRewardPlugin, PiStar0_6AdvantagePlugin, ControlModePlugin
from reward_wrapper import create_advantage_visualization_video


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


def analyze_episode_advantages(
    dataset: WrappedRobotDataset,
    episode_id: int,
    advantage_threshold: float,
    output_dir: str = "outputs"
):
    """
    Analyze advantage values for an episode and create visualization.
    
    Args:
        dataset: WrappedRobotDataset with advantage plugin
        episode_id: Episode ID to analyze
        advantage_threshold: Threshold value (middle point in graph)
        output_dir: Directory to save output videos
    """
    
    # Get episode information - handle episode subsetting
    base_dataset = dataset._datasets[0]
    
    # If using episode subsetting, we need to check if episode_id is in the subset
    if hasattr(base_dataset, 'episodes') and base_dataset.episodes is not None:
        if episode_id not in base_dataset.episodes:
            available = base_dataset.episodes[:10]  # Show first 10
            raise ValueError(
                f"Episode {episode_id} not in the selected episode subset. "
                f"Available episodes (first 10): {available}"
            )
        # Find position in subset for accessing episode_data_index
        episode_pos = base_dataset.episodes.index(episode_id)
    else:
        episode_pos = episode_id
    
    episode_length = base_dataset.meta.episodes[episode_id]["length"]
    episode_start_idx = base_dataset.episode_data_index["from"][episode_pos].item()
    
    print(f"Analyzing episode {episode_id} with {episode_length} frames")
    print(f"Advantage threshold: {advantage_threshold:.4f}")
    print(f"Advantage range: [{advantage_threshold - 1:.4f}, {advantage_threshold + 1:.4f}]")
    
    # Collect advantage data and images
    advantage_data = []
    advantage_images = []
    
    print("Processing frames...")
    for frame_idx in tqdm(range(episode_length)):
        global_idx = episode_start_idx + frame_idx
        
        # Get frame with advantage
        frame = dataset[global_idx]
        advantage_value = frame["advantage"].item()
        
        # Store advantage data
        advantage_data.append({
            'step': frame_idx,
            'advantage': advantage_value,
            'raw_advantage': advantage_value,  # Store raw value for statistics
        })
        
        # Extract images for visualization
        images = extract_images_from_frame(frame)
        advantage_images.append(images)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualization video
    output_filename = os.path.join(output_dir, f"advantage_visualization_ep_{episode_id}.mp4")
    
    print(f"Creating advantage visualization video...")
    create_advantage_visualization_video(
        advantage_images, 
        advantage_data, 
        output_filename, 
        fps=dataset.fps,
        advantage_threshold=0
    )
    
    # Calculate statistics
    raw_advantages = [d['raw_advantage'] for d in advantage_data]
    
    # Count positive/negative advantages (based on binarized values)
    binarized_advantages = [d['advantage'] for d in advantage_data]
    num_positive = sum(1 for a in binarized_advantages if a > 0)
    num_negative = sum(1 for a in binarized_advantages if a <= 0)
    
    print(f"\nAdvantage Statistics for Episode {episode_id}:")
    print(f"  Threshold: {advantage_threshold:.4f}")
    print(f"  Positive frames: {num_positive}/{episode_length} ({num_positive/episode_length*100:.1f}%)")
    print(f"  Negative frames: {num_negative}/{episode_length} ({num_negative/episode_length*100:.1f}%)")
    print(f"  Video saved to: {output_filename}")
    
    return {
        'episode_id': episode_id,
        'episode_length': episode_length,
        'threshold': advantage_threshold,
        'num_positive': num_positive,
        'num_negative': num_negative,
        'video_path': output_filename
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize pre-computed advantage values")
    parser.add_argument("--dataset-repo-id", type=str, required=True,
                        help="Repository ID of the dataset to analyze")
    parser.add_argument("--advantage-dir", type=str, required=True,
                        help="Directory containing pre-computed advantage files")
    parser.add_argument("--episode-id", type=int, required=True,
                        help="Episode ID to analyze")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save output videos")
    parser.add_argument("--percentile", type=float, default=30.0,
                        help="Percentile threshold for advantages (default: 30, meaning top 70%% are positive)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed threshold value (overrides percentile if specified)")
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset_repo_id}")
    
    # Verify advantage directory exists
    advantage_path = Path(args.advantage_dir)
    if not advantage_path.exists():
        raise FileNotFoundError(f"Advantage directory not found: {advantage_path}")
    
    # Load dataset
    base_dataset = LeRobot21Dataset(args.dataset_repo_id)
    print(f"Dataset loaded successfully. Total episodes: {base_dataset.num_episodes}")
    
    # Create advantage plugin
    if args.threshold is not None:
        # Use fixed threshold
        advantage_plugin = PiStar0_6AdvantagePlugin(
            advantage_file=args.advantage_dir,
            advantage_threshold=args.threshold,
            use_percentile_threshold=False
        )
        threshold = args.threshold
    else:
        # Use percentile threshold
        advantage_plugin = PiStar0_6AdvantagePlugin(
            advantage_file=args.advantage_dir,
            use_percentile_threshold=True,
            percentile=args.percentile
        )
        # We'll get the actual threshold from the plugin instance
        threshold = None
    
    # Wrap with required plugins
    dataset = WrappedRobotDataset(
        base_dataset, 
        plugins=[
            EpisodeOutcomePlugin(),
            ControlModePlugin(),
            PiStar0_6CumulativeRewardPlugin(normalise=True),
            advantage_plugin
        ]
    )
    
    # Get the actual threshold from the plugin instance
    if threshold is None:
        advantage_plugin_instance = dataset._plugin_instances[0][3]  # Last plugin
        threshold = advantage_plugin_instance.advantage_threshold
    
    # Analyze the episode
    print(f"\n{'='*80}")
    print(f"Processing Episode {args.episode_id}")
    print(f"{'='*80}")
    
    stats = analyze_episode_advantages(
        dataset,
        args.episode_id,
        threshold,
        args.output_dir
    )
    
    print(f"\nVisualization complete!")
    print(f"Video saved to: {stats['video_path']}")


if __name__ == "__main__":
    main()

