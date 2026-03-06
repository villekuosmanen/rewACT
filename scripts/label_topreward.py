#!/usr/bin/env python
"""
Label a LeRobot dataset with TOPReward progress scores.

Implements the TOPReward method from Chen et al. (2026):
  - For each episode, sample K video prefixes of increasing length
  - Feed each prefix to a video VLM with a binary completion prompt
  - Extract log p("True") as the raw reward signal
  - Normalise and interpolate to produce per-frame reward labels
  - Save to parquet files for use with LabelledRewardPlugin

Normalisation modes:
  per-episode       Per-episode min-max (Equation 2 of the paper).
  global-percentile Two-stage: collect raw log-probs for all episodes
                    first, then normalise globally using percentile
                    bounds. Each episode contributes equally (K samples
                    each). Use --from-analytics to skip the VLM pass
                    and re-normalise from a previous run's analytics.json.

Usage:
    # Stage 1+2 in one go with global normalisation
    python label_topreward.py \\
        --dataset-repo-id lerobot/my_dataset \\
        --instruction "Pick up the cube" \\
        --normalisation-mode global-percentile

    # Stage 2 only: re-normalise from existing analytics
    python label_topreward.py \\
        --dataset-repo-id lerobot/my_dataset \\
        --instruction "Pick up the cube" \\
        --normalisation-mode global-percentile \\
        --from-analytics path/to/analytics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from robocandywrapper import CANDYWRAPPER_PLUGINS_DIR
from robocandywrapper.factory import make_dataset_without_config

from rewact_tools.labelled_reward_plugin import LABELLED_REWARD_SUBDIR
from rewact_tools.control_mode_plugin import calculate_episode_data_index


TOPREWARD_PROMPT_TEMPLATE = (
    " The above video shows a robot manipulation trajectory that completes "
    "the following task: {instruction}. "
    "Decide whether the above statement is True or not. The answer is:"
)


# ---------------------------------------------------------------------------
# Model / processor helpers
# ---------------------------------------------------------------------------

def load_model_and_processor(model_name: str, device: str, dtype: torch.dtype):
    """Load the VLM and processor, auto-detecting the model family."""
    from transformers import AutoProcessor

    model_name_lower = model_name.lower()
    if "qwen3" in model_name_lower:
        from transformers import Qwen3VLForConditionalGeneration as ModelClass
    elif "qwen2" in model_name_lower or "qwen2.5" in model_name_lower:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    else:
        from transformers import AutoModelForVision2Seq as ModelClass

    print(f"Loading model: {model_name} (class: {ModelClass.__name__})")
    model = ModelClass.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


# ---------------------------------------------------------------------------
# Image / frame helpers
# ---------------------------------------------------------------------------

def detect_image_key(dataset) -> str:
    """Auto-detect the first image key in the dataset."""
    sample = dataset[0]
    for key in sorted(sample.keys()):
        if "image" in key and isinstance(sample[key], torch.Tensor):
            if sample[key].dim() >= 3:
                return key
    raise RuntimeError(
        "Could not auto-detect an image key. Use --image-key to specify one."
    )


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert a LeRobot image tensor to a PIL Image."""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    if img_tensor.dim() == 3 and img_tensor.shape[-1] in (1, 3):
        img_tensor = img_tensor.permute(2, 0, 1)
    if img_tensor.dtype != torch.uint8:
        if img_tensor.max() <= 1.0:
            img_tensor = (img_tensor * 255).clamp(0, 255).to(torch.uint8)
        else:
            img_tensor = img_tensor.clamp(0, 255).to(torch.uint8)
    return TF.to_pil_image(img_tensor)


def subsample_frames(
    frames: list[Image.Image], max_frames: int
) -> list[Image.Image]:
    """Uniformly subsample a list of frames to at most max_frames."""
    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[i] for i in indices]


# ---------------------------------------------------------------------------
# VLM input construction
# ---------------------------------------------------------------------------

def build_inputs_no_chat_template(
    processor,
    video_frames: list[Image.Image],
    instruction: str,
    fps: float = 30.0,
):
    """Build model inputs WITHOUT a chat template (paper default)."""
    prompt_text = (
        "<|vision_start|><|video_pad|><|vision_end|>"
        + TOPREWARD_PROMPT_TEMPLATE.format(instruction=instruction)
    )
    video_metadata = [{"fps": fps, "total_num_frames": len(video_frames)}]
    inputs = processor(
        text=[prompt_text],
        videos=[video_frames],
        video_metadata=video_metadata,
        return_tensors="pt",
        padding=True,
    )
    return inputs


def build_inputs_with_chat_template(
    processor,
    video_frames: list[Image.Image],
    instruction: str,
    fps: float = 30.0,
):
    """Build model inputs WITH a chat template."""
    from qwen_vl_utils import process_vision_info

    prompt_text = TOPREWARD_PROMPT_TEMPLATE.format(instruction=instruction)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_frames, "fps": fps},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    _, video_inputs = process_vision_info(messages)
    video_metadata = [{"fps": fps, "total_num_frames": len(video_frames)}]
    inputs = processor(
        text=[text],
        videos=video_inputs,
        video_metadata=video_metadata,
        return_tensors="pt",
        padding=True,
    )
    return inputs


def compute_log_prob_true(
    model,
    processor,
    video_frames: list[Image.Image],
    instruction: str,
    true_token_id: int,
    use_chat_template: bool,
    device: str,
    fps: float = 30.0,
) -> float:
    """Run a single forward pass and return log p("True")."""
    if use_chat_template:
        inputs = build_inputs_with_chat_template(processor, video_frames, instruction, fps=fps)
    else:
        inputs = build_inputs_no_chat_template(processor, video_frames, instruction, fps=fps)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    last_logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(last_logits.float(), dim=-1)
    return log_probs[true_token_id].item()


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_per_episode(raw_rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Per-episode min-max normalisation (Equation 2 of the paper)."""
    r_min = min(raw_rewards)
    r_max = max(raw_rewards)
    denom = r_max - r_min + eps
    return [(r - r_min) / denom for r in raw_rewards]


def compute_global_percentile_bounds(
    all_raw_stats: list[dict],
    tail_cutoff_pct: float = 2.0,
) -> tuple[float, float]:
    """Compute global normalisation bounds from per-episode raw log-probs.

    Each episode contributes its K raw samples equally (since K is constant
    across episodes, every episode has equal weight).

    Returns:
        (p_low, p_high): the lower and upper percentile bounds.
    """
    per_episode_samples = [
        np.array(ep["raw_log_probs"]) for ep in all_raw_stats
    ]
    pooled = np.concatenate(per_episode_samples)
    p_low = float(np.percentile(pooled, tail_cutoff_pct))
    p_high = float(np.percentile(pooled, 100.0 - tail_cutoff_pct))
    return p_low, p_high


def normalise_global(
    raw_rewards: list[float],
    p_low: float,
    p_high: float,
    eps: float = 1e-8,
) -> list[float]:
    """Normalise raw log-probs using global percentile bounds, clamped to [0, 1]."""
    denom = p_high - p_low + eps
    return [max(0.0, min(1.0, (r - p_low) / denom)) for r in raw_rewards]


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_to_all_frames(
    prefix_lengths: np.ndarray,
    normalised_rewards: list[float],
    episode_length: int,
) -> np.ndarray:
    """Linearly interpolate sampled rewards to every frame in the episode."""
    all_frames = np.arange(episode_length)
    return np.interp(all_frames, prefix_lengths, normalised_rewards)


# ---------------------------------------------------------------------------
# Episode data index helper
# ---------------------------------------------------------------------------

_cached_episode_data_index: dict[str, torch.Tensor] | None = None


def get_episode_data_index(raw_dataset) -> dict[str, torch.Tensor]:
    """Return episode_data_index, computing and caching it if necessary."""
    global _cached_episode_data_index
    if hasattr(raw_dataset, "episode_data_index"):
        return raw_dataset.episode_data_index
    if _cached_episode_data_index is None:
        _cached_episode_data_index = calculate_episode_data_index(raw_dataset.hf_dataset)
    return _cached_episode_data_index


# ---------------------------------------------------------------------------
# Episode collection (Stage 1)
# ---------------------------------------------------------------------------

def collect_episode_raw(
    dataset,
    episode_idx: int,
    image_key: str,
    model,
    processor,
    true_token_id: int,
    instruction: str,
    num_prefix_samples: int,
    max_video_frames: int,
    use_chat_template: bool,
    device: str,
    fps: float = 30.0,
) -> dict:
    """Run VLM on an episode and return raw log-prob statistics (no normalisation)."""
    raw_dataset = dataset._datasets[0]
    episode_length = raw_dataset.meta.episodes[episode_idx]["length"]
    ep_start = get_episode_data_index(raw_dataset)["from"][episode_idx].item()

    all_pil_frames = []
    for i in range(episode_length):
        frame = dataset[ep_start + i]
        pil_img = tensor_to_pil(frame[image_key])
        all_pil_frames.append(pil_img)

    prefix_lengths = np.unique(
        np.linspace(1, episode_length, num_prefix_samples, dtype=int)
    )

    raw_log_probs: list[float] = []
    for t in tqdm(
        prefix_lengths,
        desc=f"  Episode {episode_idx} prefixes",
        leave=False,
    ):
        prefix_frames = subsample_frames(all_pil_frames[:t], max_video_frames)
        log_p = compute_log_prob_true(
            model, processor, prefix_frames, instruction,
            true_token_id, use_chat_template, device, fps=fps,
        )
        raw_log_probs.append(log_p)

    raw_arr = np.array(raw_log_probs)
    return {
        "episode": episode_idx,
        "episode_length": episode_length,
        "ep_start": ep_start,
        "num_prefix_samples": len(prefix_lengths),
        "raw_log_prob_mean": float(raw_arr.mean()),
        "raw_log_prob_std": float(raw_arr.std()),
        "raw_log_prob_min": float(raw_arr.min()),
        "raw_log_prob_max": float(raw_arr.max()),
        "raw_log_probs": [float(v) for v in raw_log_probs],
        "prefix_lengths": [int(v) for v in prefix_lengths],
    }


# ---------------------------------------------------------------------------
# Parquet generation (Stage 2)
# ---------------------------------------------------------------------------

def write_episode_parquet(
    raw_stats: dict,
    output_dir: Path,
    normalise_fn,
) -> dict:
    """Normalise one episode's raw log-probs and write a parquet file.

    Args:
        raw_stats: Raw stats dict for the episode (from collect or analytics).
        output_dir: Directory to write the parquet into.
        normalise_fn: Callable(raw_log_probs) -> normalised list[float].

    Returns:
        Summary dict with normalised reward statistics.
    """
    episode_idx = raw_stats["episode"]
    episode_length = raw_stats["episode_length"]
    ep_start = raw_stats["ep_start"]
    raw_log_probs = raw_stats["raw_log_probs"]
    prefix_lengths = np.array(raw_stats["prefix_lengths"])

    normalised = normalise_fn(raw_log_probs)
    per_frame_rewards = interpolate_to_all_frames(
        prefix_lengths - 1,
        normalised,
        episode_length,
    )

    records = [
        {
            "frame_index": ep_start + i,
            "episode_idx": episode_idx,
            "frame_index_in_episode": i,
            "reward": float(per_frame_rewards[i]),
        }
        for i in range(episode_length)
    ]
    df = pd.DataFrame(records)

    out_file = output_dir / f"episode_{episode_idx:06d}.parquet"
    df.to_parquet(out_file, index=False)

    rewards = df["reward"].values
    return {
        "episode": episode_idx,
        "mean": float(rewards.mean()),
        "min": float(rewards.min()),
        "max": float(rewards.max()),
        "final": float(rewards[-1]),
        "file": out_file.name,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Label a LeRobot dataset with TOPReward progress scores."
    )
    parser.add_argument(
        "--dataset-repo-id", type=str, required=True, help="LeRobot dataset repo ID"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Task instruction (e.g. 'Pick up the cube')",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default=None,
        help="Image key in the dataset (auto-detected if omitted)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model ID for the video VLM",
    )
    parser.add_argument(
        "--num-prefix-samples",
        type=int,
        default=80,
        help="Number of uniformly-spaced prefix lengths per episode (K)",
    )
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=128,
        help="Maximum frames sent to the VLM per prefix",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Wrap the prompt in the model's chat template "
        "(the paper recommends NOT using this)",
    )
    parser.add_argument(
        "--normalisation-mode",
        type=str,
        default="per-episode",
        choices=["per-episode", "global-percentile"],
        help="per-episode: min-max per episode (paper default). "
        "global-percentile: percentile bounds across all episodes "
        "(equal weight per episode).",
    )
    parser.add_argument(
        "--tail-cutoff-percentile",
        type=float,
        default=2.0,
        help="Percentile for tail cutoff in global-percentile mode. "
        "E.g. 2 means the 2nd and 98th percentiles are used as "
        "the 0 and 1 normalisation targets (default: 2).",
    )
    parser.add_argument(
        "--from-analytics",
        type=str,
        default=None,
        help="Path to an existing analytics.json. Skips VLM inference "
        "and re-normalises from saved raw log-probs (stage 2 only).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Specific episode indices to process (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <dataset_root>/candywrapper_plugins/labelled_rewards)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for inference"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset (with labelled rewards) to HuggingFace Hub",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # ---- Load dataset ----
    print(f"Loading dataset: {args.dataset_repo_id}")
    dataset = make_dataset_without_config(args.dataset_repo_id)
    raw_dataset = dataset._datasets[0]
    print(f"  Episodes: {dataset.num_episodes}, Total frames: {len(dataset)}")

    image_key = args.image_key or detect_image_key(dataset)
    dataset_fps = float(raw_dataset.fps)
    print(f"  Using image key: {image_key}")
    print(f"  Dataset FPS: {dataset_fps}")

    # ---- Determine output directory ----
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path(raw_dataset.root)
            / CANDYWRAPPER_PLUGINS_DIR
            / LABELLED_REWARD_SUBDIR
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {output_dir}")
    print(f"  Normalisation mode: {args.normalisation_mode}")
    if args.normalisation_mode == "global-percentile":
        print(f"  Tail cutoff percentile: {args.tail_cutoff_percentile}%")

    # ==================================================================
    # Stage 1: Collect raw log-probs (or load from analytics)
    # ==================================================================
    if args.from_analytics:
        print(f"\nLoading raw log-probs from: {args.from_analytics}")
        with open(args.from_analytics) as f:
            analytics = json.load(f)
        all_raw_stats = analytics["episodes"]
        if args.episodes is not None:
            ep_set = set(args.episodes)
            all_raw_stats = [s for s in all_raw_stats if s["episode"] in ep_set]
        print(f"  Loaded {len(all_raw_stats)} episodes from analytics")
    else:
        # ---- Load model ----
        model, processor = load_model_and_processor(
            args.model, args.device, torch_dtype
        )

        true_token_ids = processor.tokenizer.encode("True", add_special_tokens=False)
        if len(true_token_ids) != 1:
            print(
                f"WARNING: 'True' tokenizes to {len(true_token_ids)} tokens "
                f"({true_token_ids}). Using the first one."
            )
        true_token_id = true_token_ids[0]
        print(f"  'True' token ID: {true_token_id}")
        print(f"  Chat template: {'ON' if args.use_chat_template else 'OFF (paper default)'}")

        # ---- Determine episodes ----
        if args.episodes is not None:
            episode_indices = args.episodes
        else:
            episode_indices = list(range(raw_dataset.num_episodes))
        print(f"  Processing {len(episode_indices)} episodes")

        # ---- Run VLM ----
        all_raw_stats = []
        for ep_idx in tqdm(episode_indices, desc="Stage 1: VLM inference"):
            raw_stats = collect_episode_raw(
                dataset=dataset,
                episode_idx=ep_idx,
                image_key=image_key,
                model=model,
                processor=processor,
                true_token_id=true_token_id,
                instruction=args.instruction,
                num_prefix_samples=args.num_prefix_samples,
                max_video_frames=args.max_video_frames,
                use_chat_template=args.use_chat_template,
                device=args.device,
                fps=dataset_fps,
            )
            all_raw_stats.append(raw_stats)
            print(
                f"  Episode {ep_idx}: "
                f"raw_log_p=[{raw_stats['raw_log_prob_min']:.3f}, "
                f"{raw_stats['raw_log_prob_max']:.3f}]"
            )

        # ---- Save analytics ----
        analytics = {
            "config": {
                "model": args.model,
                "dataset": args.dataset_repo_id,
                "instruction": args.instruction,
                "num_prefix_samples": args.num_prefix_samples,
                "max_video_frames": args.max_video_frames,
                "use_chat_template": args.use_chat_template,
                "dataset_fps": dataset_fps,
            },
            "episodes": all_raw_stats,
        }
        analytics_file = output_dir / "analytics.json"
        with open(analytics_file, "w") as f:
            json.dump(analytics, f, indent=2)
        print(f"\nAnalytics saved to: {analytics_file}")

    # ==================================================================
    # Stage 2: Normalise and write parquet files
    # ==================================================================
    if args.normalisation_mode == "per-episode":
        normalise_fn = normalise_per_episode
    else:
        p_low, p_high = compute_global_percentile_bounds(
            all_raw_stats, tail_cutoff_pct=args.tail_cutoff_percentile
        )
        print(
            f"\nGlobal percentile bounds "
            f"(p{args.tail_cutoff_percentile} / p{100 - args.tail_cutoff_percentile}):"
            f"  p_low={p_low:.4f}  p_high={p_high:.4f}"
        )
        normalise_fn = lambda rr: normalise_global(rr, p_low, p_high)

    all_norm_stats = []
    for raw_stats in tqdm(all_raw_stats, desc="Stage 2: Normalise & write"):
        norm_stat = write_episode_parquet(raw_stats, output_dir, normalise_fn)
        all_norm_stats.append(norm_stat)
        print(
            f"  Episode {norm_stat['episode']}: "
            f"mean={norm_stat['mean']:.3f} final={norm_stat['final']:.3f}  "
            f"-> {norm_stat['file']}"
        )

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("LABELLING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Normalisation: {args.normalisation_mode}")
    if args.normalisation_mode == "global-percentile":
        print(f"  Tail cutoff: {args.tail_cutoff_percentile}%")
        print(f"  Bounds: p_low={p_low:.4f}  p_high={p_high:.4f}")
    print(f"  Episodes processed: {len(all_norm_stats)}")
    print(f"  Output directory: {output_dir}")

    if all_norm_stats:
        means = [s["mean"] for s in all_norm_stats]
        finals = [s["final"] for s in all_norm_stats]
        raw_means = [s["raw_log_prob_mean"] for s in all_raw_stats]
        raw_mins = [s["raw_log_prob_min"] for s in all_raw_stats]
        raw_maxs = [s["raw_log_prob_max"] for s in all_raw_stats]
        print(
            f"  Normalised reward across episodes: mean={np.mean(means):.3f} "
            f"(std {np.std(means):.3f})"
        )
        print(
            f"  Normalised final reward: mean={np.mean(finals):.3f} "
            f"(std {np.std(finals):.3f})"
        )
        print(f"  Raw log p('True') across episodes:")
        print(f"    mean of means: {np.mean(raw_means):.3f} (std {np.std(raw_means):.3f})")
        print(f"    range: [{np.min(raw_mins):.3f}, {np.max(raw_maxs):.3f}]")

    if args.push_to_hub:
        print("\nPushing dataset to HuggingFace Hub...")
        try:
            raw_dataset.push_to_hub()
            print("Successfully pushed to Hub.")
        except Exception as e:
            print(f"Error pushing to Hub: {e}")


if __name__ == "__main__":
    main()
