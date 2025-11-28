#!/usr/bin/env python
"""ACTvantage Policy Trainer."""

import logging
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Optional

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval import eval_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
)
from robocandywrapper import WandBLogger
from robocandywrapper.plugins import EpisodeOutcomePlugin
from robocandywrapper import make_dataset

from rewact.plugins import PiStar0_6CumulativeRewardPlugin, ControlModePlugin, PiStar0_6AdvantagePlugin
from rewact.utils import make_actvantage_policy


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
    grad_scaler.scale(loss).backward()

    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    grad_scaler.update()

    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


class ACTvantageTrainer:
    """Trainer for ACTvantage policy."""

    def __init__(
        self,
        cfg: TrainPipelineConfig,
        advantage_dirs: Dict[str, Path],
        sampler_config: Optional[Dict] = None,
        advantage_percentile: float = 30.0,
        checkpoint_push_freq: Optional[int] = None,
    ):
        """
        Initialize ACTvantage trainer.

        Args:
            cfg: Training pipeline configuration
            advantage_dirs: Mapping of dataset repo_id to advantage directory path
            sampler_config: Sampler configuration dict (with episodes, weights, etc.).
                           If None, no episode filtering is applied.
            advantage_percentile: Percentile threshold for advantage filtering
            checkpoint_push_freq: Frequency to push checkpoints to Hub (in steps).
                                  If None, only pushes final model.
        """
        self.cfg = cfg
        self.advantage_dirs = advantage_dirs
        self.sampler_config = sampler_config
        self.advantage_percentile = advantage_percentile
        self.checkpoint_push_freq = checkpoint_push_freq
        self.wandb_logger = None
        self.wandb_run_url = None

    def train(self) -> Dict[str, Any]:
        """
        Run the training loop.

        Returns:
            Dictionary with training results including model_repo_id and wandb_url
        """
        cfg = self.cfg
        cfg.validate()
        logging.info(pformat(cfg.to_dict()))

        # Set up WandB logging
        if cfg.wandb.enable and cfg.wandb.project:
            self.wandb_logger = WandBLogger(cfg)
            # Get WandB URL
            try:
                import wandb
                self.wandb_run_url = wandb.run.get_url() if wandb.run else None
            except Exception:
                self.wandb_run_url = None
        else:
            self.wandb_logger = None
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

        if cfg.seed is not None:
            set_seed(cfg.seed)

        # Force pyav video decoding to avoid issues with torchcodec and GPU-accelerated video decoding in the cloud.
        cfg.dataset.video_backend = "pyav"

        # Apply sampler config if provided
        if self.sampler_config is not None:
            cfg.dataset.episodes = self.sampler_config.episodes

        # Check device is available
        device = get_safe_torch_device(cfg.policy.device, log=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        logging.info("Creating dataset with advantage plugin")
        
        # Validate all advantage directories exist
        missing_dirs = [repo_id for repo_id, path in self.advantage_dirs.items() if not path.exists()]
        if missing_dirs:
            logging.warning(
                f"Advantage directories not found for datasets: {missing_dirs}. "
                "Please run precompute_advantage.py first for each dataset"
            )
            raise FileNotFoundError(f"Missing advantage directories for: {missing_dirs}")
        
        advantage_plugin = PiStar0_6AdvantagePlugin(
            advantage_file=self.advantage_dirs,
            use_percentile_threshold=True,
            percentile=self.advantage_percentile,
        )

        dataset = make_dataset(
            cfg, 
            plugins=[
                EpisodeOutcomePlugin(), 
                ControlModePlugin(), 
                PiStar0_6CumulativeRewardPlugin(normalise=True), 
                advantage_plugin
            ]
        )

        # Create environment for evaluation
        eval_env = None
        if cfg.eval_freq > 0 and cfg.env is not None:
            logging.info("Creating env")
            eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

        logging.info("Creating policy")
        policy = make_actvantage_policy(cfg.policy, dataset.meta)

        logging.info("Creating optimizer and scheduler")
        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
        grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

        step = 0

        if cfg.resume:
            step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())

        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

        # Create dataloader
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
            num_workers=4,
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
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
            is_checkpoint_push_step = (
                self.checkpoint_push_freq is not None 
                and step % self.checkpoint_push_freq == 0
            )

            if is_log_step:
                logging.info(train_tracker)
                if self.wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    self.wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if cfg.save_checkpoint and is_saving_step:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)
                if self.wandb_logger:
                    self.wandb_logger.log_policy(checkpoint_dir)

                # Push checkpoint to Hub if configured
                if is_checkpoint_push_step and cfg.policy.push_to_hub:
                    logging.info(f"Pushing checkpoint {step} to Hub")
                    self._push_checkpoint_to_hub(policy, cfg, step)

            if cfg.env and is_eval_step:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
                    eval_info = eval_policy(
                        eval_env,
                        policy,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(eval_tracker)
                if self.wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    self.wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    self.wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

        if eval_env:
            eval_env.close()
        logging.info("End of training")

        # Push final model to Hub
        model_repo_id = None
        if cfg.policy.push_to_hub:
            # Format datasets properly for YAML frontmatter
            if cfg.dataset.repo_id.startswith('[') and cfg.dataset.repo_id.endswith(']'):
                datasets_str = cfg.dataset.repo_id.strip('[]')
                datasets = [ds.strip('\'\" ') for ds in datasets_str.split(',')]
                cfg.dataset.repo_id = datasets
            policy.push_model_to_hub(cfg)
            model_repo_id = cfg.policy.repo_id
            logging.info(f"Model pushed to Hub: {model_repo_id}")

        return {
            "model_repo_id": model_repo_id,
            "wandb_url": self.wandb_run_url,
            "final_loss": train_tracker.loss.avg,
            "total_steps": step,
        }

    def _push_checkpoint_to_hub(self, policy, cfg, step):
        """Push intermediate checkpoint to Hub."""
        try:
            # Create a checkpoint-specific repo ID
            base_repo_id = cfg.policy.repo_id
            checkpoint_repo_id = f"{base_repo_id}-checkpoint-{step}"
            
            # Temporarily modify config for this checkpoint
            original_repo_id = cfg.policy.repo_id
            cfg.policy.repo_id = checkpoint_repo_id
            
            policy.push_model_to_hub(cfg)
            logging.info(f"Pushed checkpoint to {checkpoint_repo_id}")
            
            # Restore original repo ID
            cfg.policy.repo_id = original_repo_id
        except Exception as e:
            logging.warning(f"Failed to push checkpoint to Hub: {e}")

