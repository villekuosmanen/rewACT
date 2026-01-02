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
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

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

from rewact_tools import PiStar0_6CumulativeRewardPlugin, ControlModePlugin
from rewact_tools import make_pre_post_processors
from scripts.utils import make_actvantage_policy


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
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    sampler_config = load_sampler_config("scripts/configs/sampler_actvantage.json")
    cfg.dataset.episodes = sampler_config.episodes

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    from rewact_tools import PiStar0_6AdvantagePlugin
    
    # For multiple dataset training, provide a mapping of repo_id to advantage directory:
    advantage_dirs = {
        "villekuosmanen/build_block_tower": Path("outputs/build_block_tower-advantages"),
        "villekuosmanen/dAgger_build_block_tower_1.0.0": Path("outputs/dAgger_build_block_tower_1.0.0-advantages"),
        "villekuosmanen/dAgger_build_block_tower_1.1.0": Path("outputs/dAgger_build_block_tower_1.1.0-advantages"),
        "villekuosmanen/dAgger_build_block_tower_1.2.0": Path("outputs/dAgger_build_block_tower_1.2.0-advantages"),
        "villekuosmanen/dAgger_build_block_tower_1.3.0": Path("outputs/dAgger_build_block_tower_1.3.0-advantages"),
        "villekuosmanen/dAgger_build_block_tower_1.4.0": Path("outputs/dAgger_build_block_tower_1.4.0-advantages"),
    }
    
    # Validate all directories exist
    missing_dirs = [repo_id for repo_id, path in advantage_dirs.items() if not path.exists()]
    if missing_dirs:
        logging.warning(
            f"Advantage directories not found for datasets: {missing_dirs}. "
            "Please run precompute_advantage.py first for each dataset"
        )
        raise FileNotFoundError(f"Missing advantage directories for: {missing_dirs}")
    
    advantage_plugin = PiStar0_6AdvantagePlugin(
        advantage_file=advantage_dirs,  # Now accepts dict for multiple datasets
        use_percentile_threshold=True,
        percentile=55.0,  # Can be configured
    )
    dataset = make_dataset(cfg, plugins=[EpisodeOutcomePlugin(), ControlModePlugin(), PiStar0_6CumulativeRewardPlugin(normalise=True), advantage_plugin])
    import numpy as np
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
    policy = make_actvantage_policy(cfg.policy, dataset.meta)

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

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
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=False,
        multiprocessing_context='spawn',
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

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
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
