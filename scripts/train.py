#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RewACT Training Script - Train reward-augmented ACT policies on LeRobot datasets.

Quickstart:
    python scripts/train.py \
        --dataset.repo_id=danaaubakirova/so100_task_2 \
        --dataset.episodes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] \
        --policy.type=rewact \
        --policy.repo_id=your-hf-user/so100_rewact_resnet \
        --batch_size=8 --steps=1000 --save_freq=500

With DINOv3 backbone:
    python scripts/train.py \
        --dataset.repo_id=danaaubakirova/so100_task_2 \
        --dataset.episodes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] \
        --policy.type=rewact \
        --policy.repo_id=your-hf-user/so100_rewact_dinov3 \
        --batch_size=1 --steps=1000 --save_freq=500 \
        --policy.vision_encoder_type=dinov3 \
        --policy.dinov3.variant=vitb16 \
        --policy.dinov3.weights=/path/to/dinov3_vitb16.pth \
        --policy.dinov3.use_patch_merge=True

job_name and output_dir default to the model name from policy.repo_id.
See README.md for all options.
"""

import logging
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from robocandywrapper import WandBLogger
from robocandywrapper.plugins import EpisodeOutcomePlugin
from robocandywrapper.samplers import load_sampler_config
from robocandywrapper import make_dataset

from rewact.plugins import PiStar0_6CumulativeRewardPlugin, ControlModePlugin
from rewact.utils import make_rewact_policy
from rewact.policies.factory import make_pre_post_processors


def _check_hf_write_access(repo_id: str) -> None:
    """Check if we have write access to the HuggingFace repo before training."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
    
    if not repo_id:
        return
    
    api = HfApi()
    try:
        # Check if user is logged in
        user_info = api.whoami()
        username = user_info.get("name", "")
    except Exception:
        raise RuntimeError(
            "Not logged in to HuggingFace. Run `huggingface-cli login` first."
        )
    
    repo_owner = repo_id.split("/")[0]
    
    # Check if repo exists and we have write access
    try:
        repo_info = api.repo_info(repo_id, repo_type="model")
        # Repo exists - check if we can write to it
        # If we own it or are a collaborator, we're good
        logging.info(f"HuggingFace repo '{repo_id}' exists and is accessible.")
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            # Repo doesn't exist - check if we can create it
            if repo_owner != username:
                # Check if it's an org we belong to
                orgs = [org.get("name", "") for org in user_info.get("orgs", [])]
                if repo_owner not in orgs:
                    raise RuntimeError(
                        f"Cannot create repo '{repo_id}': you are logged in as '{username}' "
                        f"but the repo owner '{repo_owner}' is not you or an org you belong to."
                    )
            logging.info(f"HuggingFace repo '{repo_id}' will be created on push.")
        else:
            raise RuntimeError(f"Cannot access HuggingFace repo '{repo_id}': {e}")


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    # Default job_name and output_dir from policy.repo_id if not set
    if hasattr(cfg.policy, 'repo_id') and cfg.policy.repo_id:
        run_name = cfg.policy.repo_id.split('/')[-1]
        if not cfg.job_name:
            cfg.job_name = run_name
        if not cfg.output_dir:
            cfg.output_dir = Path(f"outputs/train/{run_name}")
    
    # Early check for HuggingFace write access when repo_id is provided
    if hasattr(cfg.policy, 'repo_id') and cfg.policy.repo_id:
        _check_hf_write_access(cfg.policy.repo_id)
    
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Force pyav video decoding to avoid issues with torchcodec and GPU-accelerated video decoding in the cloud.
    # cfg.dataset.video_backend = "pyav"

    sampler_config = load_sampler_config("scripts/configs/sampler_rewact.json")
    cfg.dataset.episodes = sampler_config.episodes

    # Check device is available
    device_str = cfg.policy.device if cfg.policy.device is not None else "cuda"
    device = get_safe_torch_device(device_str, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg, plugins=[EpisodeOutcomePlugin(), ControlModePlugin(), PiStar0_6CumulativeRewardPlugin(normalise=True)])
    dataset.meta.features['observation.eef_6d_pose']= {
        'dtype': "float32",
        'shape': (7,),
    }
    # Update stats to match the new shape by appending the last element from observation.state
    for stat_key in ['min', 'max', 'mean', 'std']:
        dataset.meta.stats['observation.eef_6d_pose'][stat_key] = np.concatenate([
            dataset.meta.stats['observation.eef_6d_pose'][stat_key],
            dataset.meta.stats['observation.state'][stat_key][-1:]
        ])

    logging.info("Creating policy")
    policy = make_rewact_policy(cfg.policy, dataset.meta)

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    # TODO: make sure it handles plugin features
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        plugin_features=dataset.plugin_features,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Benchmarking data loading performance
    # Test single batch load speed
    import time
    test_dataset = dataset
    print(f"Dataset length: {len(test_dataset)}")
    start = time.perf_counter()
    sample = test_dataset[0]
    print(f"Single sample load: {time.perf_counter() - start:.3f}s")
    start = time.perf_counter()
    sample = test_dataset[0]
    print(f"Same sample load twice: {time.perf_counter() - start:.3f}s")
    start = time.perf_counter()
    sample = test_dataset[1000]
    print(f"Second sample load: {time.perf_counter() - start:.3f}s")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                cfg=cfg,
                policy=policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

    logging.info("End of training")

    if cfg.policy.push_to_hub:
        # Format datasets properly for YAML frontmatter
        if cfg.dataset.repo_id.startswith('[') and cfg.dataset.repo_id.endswith(']'):
            # Handle multiple datasets: "[dataset1, dataset2]" -> ["dataset1", "dataset2"]
            datasets_str = cfg.dataset.repo_id.strip('[]')
            datasets = [ds.strip('\'\" ') for ds in datasets_str.split(',')]
            cfg.dataset.repo_id = datasets
        policy.push_model_to_hub(cfg)
        preprocessor.push_to_hub(cfg.policy.repo_id)
        postprocessor.push_to_hub(cfg.policy.repo_id)

def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
