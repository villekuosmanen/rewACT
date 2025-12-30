#!/usr/bin/env python

"""
Visualize reward predictions from a trained RewACT policy.

Quickstart:
    python scripts/visualise_reward_predictions.py \
        --dataset-repo-id "danaaubakirova/so100_task_2" \
        --episode-id 24 \
        --policy-path "outputs/train/so100_rewact_resnet/checkpoints/last/pretrained_model" \
        --output "outputs/eval_resnet_ep24.mp4"

"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict

import draccus
import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robocandywrapper.dataformats.lerobot_21 import LeRobot21Dataset

from reward_wrapper import ACTPolicyWithReward, create_reward_frame, create_video_from_frames
from rewact.dataset_with_reward import LeRobotDatasetWithReward
from rewact.utils import make_rewact_policy
from rewact.policies.factory import make_pre_post_processors


def load_policy(policy_path: str, dataset, dataset_meta, policy_overrides: list = None) -> Tuple[torch.nn.Module, dict]:
    """Load and initialize a policy from checkpoint."""
    
    # Load regular LeRobot policy
    if policy_overrides:
        # Convert list of "key=value" strings to dict
        overrides = {}
        for override in policy_overrides:
            key, value = override.split('=', 1)
            overrides[key] = value
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path, **overrides)
    else:
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.pretrained_path = policy_path

    # Create RewACT policy using the utility function
    policy = make_rewact_policy(policy_cfg, dataset_meta)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=dataset_meta.stats,
        plugin_features={},
    )
        
    return policy, policy_cfg, preprocessor, postprocessor


def prepare_observation(frame: dict, device: torch.device) -> dict:
    """Convert dataset frame to policy observation format."""
    observation = {}
    
    for key, value in frame.items():
        if "image" in key:
            if isinstance(value, torch.Tensor):
                # Squeeze extra dims, convert HWC -> CHW
                while value.dim() > 3:
                    value = value.squeeze(0)
                if value.shape[-1] in [1, 3]:  # HWC format
                    value = value.permute(2, 0, 1)
                value = value.float()
                if value.max() > 1.0:
                    value = value / 255.0
            observation[key] = value.unsqueeze(0).to(device)
            
        elif "state" in key:
            if not isinstance(value, torch.Tensor):
                value = torch.from_numpy(value).float()
            observation[key] = value.unsqueeze(0).to(device)
    
    return observation

def analyze_episode(
    dataset: LeRobotDataset,
    policy,
    preprocessor,
    postprocessor,
    output_path: str,
    episode_id: int,
    device: torch.device,
    model_dtype: torch.dtype = torch.float32
) -> Dict:
    """
    Run policy inference on an episode and analyze proprioceptive importance.
    
    Returns:
        Dictionary containing analysis results
    """
    
    # Filter dataset to only include the specified episode
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_id)
    
    if len(episode_frames) == 0:
        raise ValueError(f"Episode {episode_id} not found")
    
    print(f"Analyzing episode {episode_id} ({len(episode_frames)} frames)")
    
    reward_data = []
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for i in tqdm(range(len(episode_frames)), desc="Processing"):
            frame = dataset[episode_frames[i]['index'].item()]
            observation = prepare_observation(frame, device)
            
            with torch.inference_mode():
                _, reward = policy.select_action(observation)
                reward_data.append({'step': i, 'reward': reward})
                
                # Extract images for visualization
                images = []
                for key in observation:
                    if "image" in key and "past" not in key:
                        img = observation[key].squeeze(0) * 255
                        img = img.permute(1, 2, 0).cpu()
                        images.append(img)
                
                # Write frame immediately to disk
                frame_path = temp_path / f"frame_{i:06d}.png"
                create_reward_frame(images, reward_data[-1], reward_data, frame_path, (640, 480), 200, total_steps=len(episode_frames))
            
        
        # Create video from frames
        print("Creating video...")
        create_video_from_frames(temp_path, output_path, fps=dataset.fps)
    
    # Print stats
    rewards = [r['reward'] for r in reward_data]
    print(f"Reward: mean={np.mean(rewards):.3f}, min={np.min(rewards):.3f}, max={np.max(rewards):.3f}, final={rewards[-1]:.3f}")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize RewACT reward predictions")
    parser.add_argument("--dataset-repo-id", type=str, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--policy-overrides", type=str, nargs="*")
    parser.add_argument("--output", type=str, default="outputs/reward_visualization.mp4")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Loading dataset: {args.dataset_repo_id}")
    
    # Load dataset
    dataset = LeRobot21Dataset(args.dataset_repo_id)
    print(f"Dataset loaded successfully. Total episodes: {dataset.num_episodes}")

    # Determine which episodes to analyze
    if args.episode_id is not None:
        # Single episode analysis
        if args.episode_id >= dataset.num_episodes:
            raise ValueError(f"Episode {args.episode_id} not found. Dataset has {dataset.num_episodes} episodes.")
        episodes_to_analyze = [args.episode_id]
        print(f"Target episode: {args.episode_id}")
    else:
        # All episodes analysis
        episodes_to_analyze = list(range(dataset.num_episodes))
        print(f"Will analyze all {dataset.num_episodes} episodes")
    
    print("Loading policy...")
    # TODO this currently will not work
    policy, _, preprocessor, postprocessor = load_policy(
        args.policy_path,
        dataset,
        dataset.meta,
        args.policy_overrides
    )
    
    # Wrap dataset (with temporal_offset for VJEPA2)
    temporal_offset = 0
    if policy.config.vision_encoder_type == "vjepa2":
        temporal_offset = getattr(policy.config, 'temporal_offset', 30)
    dataset = LeRobotDatasetWithReward(base_dataset, temporal_offset=temporal_offset)
    
    for episode_id in tqdm(episodes_to_analyze, desc="Analyzing episodes"):
        print(f"\nStarting analysis of episode {episode_id}...")
        analyze_episode(
            dataset=dataset,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            episode_id=episode_id,
            device=device,
            model_dtype=model_dtype
        )
        print(f"Episode {episode_id} analysis completed successfully")
            
if __name__ == "__main__":
    main()
