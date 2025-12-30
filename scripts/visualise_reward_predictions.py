#!/usr/bin/env python

"""
Visualize reward predictions from a trained RewACT policy.

Quickstart:
    python scripts/visualise_reward_predictions.py \
        --dataset-repo-id "danaaubakirova/so100_task_2" \
        --episode-id 24 \
        --policy-path "outputs/train/so100_rewact_resnet/checkpoints/last/pretrained_model"

Output: outputs/reward_visualization.mp4
"""

import argparse
import json
import tempfile
from pathlib import Path

import draccus
import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from reward_wrapper import ACTPolicyWithReward, create_reward_frame, create_video_from_frames
from rewact.dataset_with_reward import LeRobotDatasetWithReward
from rewact.utils import make_rewact_policy


def load_policy(policy_path: str, dataset_meta, policy_overrides: list = None):
    """Load a RewACT policy from local path or HuggingFace."""
    policy_path = Path(policy_path).resolve()
    
    if policy_path.exists():
        # Local path
        config_file = policy_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file) as f:
            config_dict = json.load(f)
        
        if policy_overrides:
            for override in policy_overrides:
                key, value = override.split('=', 1)
                config_dict[key] = value
        
        config_type = config_dict.pop("type", "rewact")
        config_class = PreTrainedConfig.get_choice_class(config_type)
        policy_cfg = draccus.decode(config_class, config_dict)
        policy_cfg.pretrained_path = str(policy_path)
    else:
        # HuggingFace repo ID
        overrides = {}
        if policy_overrides:
            for override in policy_overrides:
                key, value = override.split('=', 1)
                overrides[key] = value
        policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path), **overrides)
        policy_cfg.pretrained_path = str(policy_path)

    policy = make_rewact_policy(policy_cfg, dataset_meta)
    return ACTPolicyWithReward(policy, recording_enabled=False), policy_cfg


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


def analyze_episode(dataset: LeRobotDataset, policy, episode_id: int, device: torch.device):
    """Run policy on an episode and create reward visualization."""
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_id)
    
    if len(episode_frames) == 0:
        raise ValueError(f"Episode {episode_id} not found")
    
    print(f"Analyzing episode {episode_id} ({len(episode_frames)} frames)")
    
    reward_data = []
    output_file = "outputs/reward_visualization.mp4"
    Path("outputs").mkdir(exist_ok=True)
    
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
        create_video_from_frames(temp_path, output_file, fps=dataset.fps)
    
    # Print stats
    rewards = [r['reward'] for r in reward_data]
    print(f"Reward: mean={np.mean(rewards):.3f}, min={np.min(rewards):.3f}, max={np.max(rewards):.3f}, final={rewards[-1]:.3f}")
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize RewACT reward predictions")
    parser.add_argument("--dataset-repo-id", type=str, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--policy-overrides", type=str, nargs="*")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Loading dataset: {args.dataset_repo_id}")
    
    base_dataset = LeRobotDataset(args.dataset_repo_id)
    print(f"Dataset: {base_dataset.num_episodes} episodes")
    
    if args.episode_id >= base_dataset.num_episodes:
        raise ValueError(f"Episode {args.episode_id} not found (max: {base_dataset.num_episodes - 1})")
    
    print("Loading policy...")
    policy, _ = load_policy(args.policy_path, base_dataset.meta, args.policy_overrides)
    
    # Wrap dataset (with temporal_offset for VJEPA2)
    temporal_offset = 0
    if policy.config.vision_encoder_type == "vjepa2":
        temporal_offset = getattr(policy.config, 'temporal_offset', 30)
    dataset = LeRobotDatasetWithReward(base_dataset, temporal_offset=temporal_offset)
    
    policy.eval()
    policy.to(device)
    
    analyze_episode(dataset, policy, args.episode_id, device)


if __name__ == "__main__":
    main()
