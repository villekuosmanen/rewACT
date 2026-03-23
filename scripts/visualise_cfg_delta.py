#!/usr/bin/env python

"""
Visualise the classifier-free guidance (CFG) delta for ACTVantage policy.

For each frame in an episode, runs the policy twice — once with advantage=0
(unconditional) and once with advantage=1 (conditional) — and reports the
per-joint delta across the predicted action chunk.

Usage:
    python visualise_cfg_delta.py \
        --dataset-repo-id villekuosmanen/build_block_tower \
        --policy-path checkpoints/actvantage_run1/pretrained_model \
        --episode-id 0 \
        --gamma 3.0
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from robocandywrapper import make_dataset_without_config

from rewact_tools import make_pre_post_processors
from utils import make_actvantage_policy


ACTION_DIM_NAMES = [
    "shoulder_rotate", "shoulder_lift", "elbow_flex",
    "forearm_roll", "wrist_angle", "wrist_rotate",
    "gripper",
]


def load_policy(policy_path: str, dataset_meta):
    """Load an ACTVantage policy and its processors from a checkpoint."""
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path

    policy = make_actvantage_policy(policy_cfg, dataset_meta)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=dataset_meta.stats,
        plugin_features={},
    )
    return policy, preprocessor, postprocessor


def prepare_observation(frame: dict, device: torch.device) -> dict:
    """Convert a dataset frame into a policy-ready observation dict."""
    observation = {}
    for key, value in frame.items():
        if "image" not in key:
            continue
        if not isinstance(value, torch.Tensor):
            continue

        while value.dim() > 3:
            value = value.squeeze(0)

        h, w, c = value.shape
        if c in (1, 3):
            value = value.permute(2, 0, 1)
        elif h in (1, 3):
            pass
        else:
            value = value.permute(2, 0, 1)

        value = value.float()
        if value.max() > 1.0:
            value = value / 255.0

        observation[key] = value.unsqueeze(0).to(device)

    for key in ("observation.state", "observation.state.pos"):
        if key in frame:
            val = frame[key]
            if not isinstance(val, torch.Tensor):
                val = torch.from_numpy(val).float()
            observation["observation.state"] = val.unsqueeze(0).to(device)
            break

    return observation


def run_episode(
    dataset,
    policy,
    preprocessor,
    episode_id: int,
    gamma: float,
    device: torch.device,
):
    """Run CFG inference on every frame of an episode and return per-joint deltas."""
    episode_frames = dataset._datasets[0].hf_dataset.filter(
        lambda x: x["episode_index"] == episode_id
    )
    episode_length = len(episode_frames)
    if episode_length == 0:
        raise ValueError(f"Episode {episode_id} not found or is empty")

    print(f"Episode {episode_id}: {episode_length} frames")

    all_deltas = []

    policy.eval()
    for i in tqdm(range(episode_length), desc="Frames"):
        frame = dataset[episode_frames[i]["index"].item()]
        if "observation.state.pos" in frame:
            frame["observation.state"] = frame["observation.state.pos"]
        if "action.pos" in frame:
            frame["action"] = frame["action.pos"]

        obs = prepare_observation(frame, device)

        with torch.inference_mode():
            batch = preprocessor(obs)

            actions_uncond = policy.predict_action_chunk(batch, advantage_value=0.0)
            actions_cond = policy.predict_action_chunk(batch, advantage_value=1.0)
            delta = actions_cond - actions_uncond  # (1, chunk_size, action_dim)

            all_deltas.append(delta.squeeze(0).cpu().numpy())

    return np.array(all_deltas)  # (T, chunk_size, action_dim)


def print_report(deltas: np.ndarray, gamma: float, episode_id: int):
    """Print a per-joint summary of the CFG delta."""
    T, chunk_size, action_dim = deltas.shape

    if action_dim == len(ACTION_DIM_NAMES):
        dim_names = ACTION_DIM_NAMES
    else:
        dim_names = [f"joint_{j}" for j in range(action_dim)]

    # Use the first action step in each chunk for the summary
    first_step_deltas = deltas[:, 0, :]  # (T, action_dim)
    scaled = first_step_deltas * gamma

    print(f"\n{'=' * 72}")
    print(f"CFG Delta Report  —  episode {episode_id}  |  gamma = {gamma}")
    print(f"{'=' * 72}")
    print(f"{'Joint':<16} {'mean delta':>12} {'std delta':>12} "
          f"{'|mean|':>10} {'max |delta|':>12}")
    print(f"{'-' * 72}")

    for j, name in enumerate(dim_names):
        col = scaled[:, j]
        print(
            f"{name:<16} {col.mean():>12.5f} {col.std():>12.5f} "
            f"{np.abs(col).mean():>10.5f} {np.abs(col).max():>12.5f}"
        )

    print(f"{'-' * 72}")
    overall = np.abs(scaled).mean(axis=0)
    print(f"{'ALL (|mean|)':<16} {overall.mean():>12.5f}")
    print()

    # Per-frame evolution: show every ~10 % of the episode
    stride = max(1, T // 10)
    print(f"Per-frame |delta| (gamma-scaled, first action step):")
    print(f"{'frame':>6}  " + "  ".join(f"{n:>10}" for n in dim_names))
    for t in range(0, T, stride):
        vals = np.abs(scaled[t])
        print(f"{t:>6}  " + "  ".join(f"{v:>10.5f}" for v in vals))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Visualise per-joint CFG delta for ACTVantage policy"
    )
    parser.add_argument(
        "--dataset-repo-id", type=str, required=True,
        help="HuggingFace repo ID of the dataset",
    )
    parser.add_argument(
        "--policy-path", type=str, required=True,
        help="Path to ACTVantage checkpoint directory",
    )
    parser.add_argument(
        "--episode-id", type=int, default=0,
        help="Episode to analyse (default: 0)",
    )
    parser.add_argument(
        "--gamma", type=float, default=3.0,
        help="CFG guidance scale (default: 3.0)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--output-npy", type=str, default=None,
        help="Optional path to save the raw delta array (T, chunk, action_dim)",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}")

    print(f"Loading dataset: {args.dataset_repo_id}")
    dataset = make_dataset_without_config(
        args.dataset_repo_id,
        key_rename_map={
            "action.pos": "action",
            "observation.state.pos": "observation.state",
        },
    )
    print(f"Loaded — {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    print(f"Loading policy: {args.policy_path}")
    policy, preprocessor, _ = load_policy(args.policy_path, dataset.meta)
    policy.to(device)
    print("Policy ready")

    deltas = run_episode(
        dataset, policy, preprocessor,
        episode_id=args.episode_id,
        gamma=args.gamma,
        device=device,
    )

    print_report(deltas, gamma=args.gamma, episode_id=args.episode_id)

    if args.output_npy:
        np.save(args.output_npy, deltas)
        print(f"Raw deltas saved to {args.output_npy}")


if __name__ == "__main__":
    main()
