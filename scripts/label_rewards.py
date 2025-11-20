#!/usr/bin/env python

"""
Script to label rewards from a JSON configuration file.
Loads reward specifications and applies them to episodes in a LeRobotDataset.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robocandywrapper import WrappedRobotDataset

from rewact import RewardPlugin, RewardPluginInstance, KeypointReward, get_plugin_instance


def load_reward_config(config_path: str) -> Dict:
    """
    Load reward configuration from JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Dictionary containing reward configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Reward configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    if "repo_id" not in config:
        raise ValueError("Configuration must contain 'repo_id' field")
    
    if "episodes" not in config:
        raise ValueError("Configuration must contain 'episodes' field")
    
    return config


def parse_episode_keypoints(episode_config: Dict) -> List[KeypointReward]:
    """
    Parse keypoints from episode configuration.
    
    Args:
        episode_config: Episode configuration dictionary
        
    Returns:
        List of KeypointReward objects
    """
    if "keypoints" not in episode_config:
        return []
    
    keypoints = []
    for kp_config in episode_config["keypoints"]:
        # Support both timestamp and frame_index based keypoints
        if "timestamp" in kp_config:
            keypoint = KeypointReward(
                reward=kp_config["reward"],
                timestamp=kp_config["timestamp"]
            )
        elif "frame_index" in kp_config:
            keypoint = KeypointReward(
                reward=kp_config["reward"],
                frame_index=kp_config["frame_index"]
            )
        else:
            raise ValueError(f"Keypoint must specify either 'timestamp' or 'frame_index': {kp_config}")
        
        keypoints.append(keypoint)
    
    return keypoints


def label_episode_rewards(
    reward_dataset: WrappedRobotDataset,
    reward_plugin: RewardPluginInstance,
    episode_id: int,
    keypoints: List[KeypointReward],
    verbose: bool = True
):
    """
    Label rewards for a specific episode.
    
    Args:
        reward_dataset: WrappedRobotDataset instance
        reward_plugin: RewardPluginInstance to add rewards to
        episode_id: Episode ID to label
        keypoints: List of KeypointReward objects
        verbose: Whether to print detailed information
    """
    if verbose:
        print(f"\nLabeling episode {episode_id}:")
        
        # Get episode info from the base dataset
        base_dataset = reward_dataset._datasets[0]
        episode_length = base_dataset.meta.episodes[episode_id]["length"]
        episode_duration = episode_length / base_dataset.fps
        
        print(f"  Episode length: {episode_length} frames ({episode_duration:.1f} seconds at {base_dataset.fps} FPS)")
        print(f"  Adding {len(keypoints)} keypoints:")
        
        for i, kp in enumerate(keypoints, 1):
            if kp.timestamp is not None:
                print(f"    {i}. {kp.timestamp:.1f}s -> reward {kp.reward:.1f}")
            else:
                print(f"    {i}. frame {kp.frame_index} -> reward {kp.reward:.1f}")
    
    # Add keypoints to episode using the plugin instance
    reward_plugin.add_episode_rewards(episode_id, keypoints)
    
    if verbose:
        # Show interpolated result at a few sample points
        print(f"  Keypoints added successfully!")
        
        # Sample a few frames to show interpolated rewards
        base_dataset = reward_dataset._datasets[0]
        sample_frames = [0, episode_length // 4, episode_length // 2, 3 * episode_length // 4, episode_length - 1]
        print(f"  Sample interpolated rewards:")
        
        ep_start = base_dataset.episode_data_index["from"][episode_id].item()
        for frame_idx in sample_frames:
            global_idx = ep_start + frame_idx
            if global_idx < len(reward_dataset):
                item = reward_dataset[global_idx]
                reward_value = item["reward"].item()
                timestamp = frame_idx / base_dataset.fps
                print(f"    Frame {frame_idx} ({timestamp:.1f}s): {reward_value:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Label rewards from JSON configuration")
    parser.add_argument("--config", type=str, default="demo_rewards.json",
                        help="Path to reward configuration JSON file")
    parser.add_argument("--dataset-repo-id", type=str, default=None,
                        help="Override dataset repo ID from config file")
    parser.add_argument("--episodes", type=str, nargs="*", default=None,
                        help="Specific episode IDs to label (if not specified, labels all episodes in config)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be labeled without actually applying changes")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push the labeled dataset to Hugging Face Hub after labeling")
    parser.add_argument("--clear-existing", action="store_true",
                        help="Clear existing reward labels before adding new ones")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed information")
    
    args = parser.parse_args()
    
    print(f"Loading reward configuration from: {args.config}")
    
    # Load configuration
    try:
        config = load_reward_config(args.config)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Get dataset repo ID
    repo_id = args.dataset_repo_id or config["repo_id"]
    print(f"Dataset repo ID: {repo_id}")
    
    # Determine episodes to process
    available_episodes = list(config["episodes"].keys())
    if args.episodes:
        episodes_to_process = args.episodes
        # Validate that requested episodes exist in config
        missing_episodes = set(episodes_to_process) - set(available_episodes)
        if missing_episodes:
            print(f"Warning: Episodes {missing_episodes} not found in configuration")
            episodes_to_process = [ep for ep in episodes_to_process if ep in available_episodes]
    else:
        episodes_to_process = available_episodes
    
    print(f"Episodes to process: {episodes_to_process}")
    
    if not episodes_to_process:
        print("No episodes to process")
        return 0
    
    if args.dry_run:
        print("\n🏃 DRY RUN MODE - No changes will be applied")
        
        for episode_id_str in episodes_to_process:
            episode_config = config["episodes"][episode_id_str]
            keypoints = parse_episode_keypoints(episode_config)
            
            print(f"\nWould label episode {episode_id_str}:")
            for i, kp in enumerate(keypoints, 1):
                if kp.timestamp is not None:
                    print(f"  {i}. {kp.timestamp:.1f}s -> reward {kp.reward:.1f}")
                else:
                    print(f"  {i}. frame {kp.frame_index} -> reward {kp.reward:.1f}")
        
        return 0
    
    # Load dataset
    print(f"\nLoading dataset: {repo_id}")
    try:
        base_dataset = LeRobotDataset(repo_id)
        print(f"Dataset loaded successfully. Total episodes: {base_dataset.num_episodes}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    # Wrap with reward plugin
    reward_plugin_obj = RewardPlugin(reward_start_pct=0.05, reward_end_pct=0.95)
    reward_dataset = WrappedRobotDataset(base_dataset, plugins=[reward_plugin_obj])
    
    # Get the reward plugin instance for the dataset
    reward_plugin_instance = get_plugin_instance(reward_dataset, RewardPlugin, dataset_idx=0)
    if reward_plugin_instance is None:
        print("Error: Could not get reward plugin instance")
        return 1
    
    # Clear existing rewards if requested
    if args.clear_existing:
        print("\nClearing existing reward labels...")
        for episode_id_str in episodes_to_process:
            episode_id = int(episode_id_str)
            if episode_id < base_dataset.num_episodes:
                reward_plugin_instance.remove_episode_rewards(episode_id)
        print("Existing rewards cleared")
    
    # Process each episode
    success_count = 0
    error_count = 0
    
    for episode_id_str in episodes_to_process:
        episode_id = int(episode_id_str)
        
        # Validate episode exists
        if episode_id >= base_dataset.num_episodes:
            print(f"Warning: Episode {episode_id} does not exist in dataset (max: {base_dataset.num_episodes - 1})")
            error_count += 1
            continue
        
        episode_config = config["episodes"][episode_id_str]
        keypoints = parse_episode_keypoints(episode_config)
        
        if not keypoints:
            print(f"Warning: No keypoints found for episode {episode_id}")
            continue
        
        label_episode_rewards(reward_dataset, reward_plugin_instance, episode_id, keypoints, args.verbose)
        success_count += 1
                
    # Summary
    print(f"\n{'='*60}")
    print("LABELING SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully labeled episodes: {success_count}")
    print(f"Errors: {error_count}")
    
    if success_count > 0:
        reward_file_path = reward_plugin_instance._get_reward_file_path()
        print(f"Reward labels saved to: {reward_file_path}")
        
        # Show some statistics
        total_keypoints = 0
        for episode_id_str in episodes_to_process:
            episode_id = int(episode_id_str)
            if episode_id < base_dataset.num_episodes:
                keypoints = reward_plugin_instance.get_episode_keypoints(episode_id)
                total_keypoints += len(keypoints)
        
        print(f"Total keypoints added: {total_keypoints}")
    
    # Push to hub if requested
    if args.push_to_hub and success_count > 0:
        print(f"\nPushing labeled dataset to Hugging Face Hub...")
        try:
            # Note: This requires the dataset to be set up for pushing
            # In practice, you might need additional authentication and permissions
            base_dataset.push_to_hub()
            print("✓ Successfully pushed to Hub")
        except Exception as e:
            print(f"Error pushing to Hub: {e}")
            print("Note: Make sure you have write permissions and proper authentication")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    main()
