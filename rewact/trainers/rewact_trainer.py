#!/usr/bin/env python
"""RewACT Value Function Trainer."""

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

from rewact.plugins import PiStar0_6CumulativeRewardPlugin
from rewact.utils import make_rewact_policy


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


class RewACTTrainer:
    """Trainer for RewACT value function."""

    def __init__(
        self,
        cfg: TrainPipelineConfig,
        sampler_config: Optional[Dict] = None,
        checkpoint_push_freq: Optional[int] = None,
    ):
        """
        Initialize RewACT trainer.

        Args:
            cfg: Training pipeline configuration
            sampler_config: Sampler configuration dict (with episodes, weights, etc.).
                           If None, no episode filtering is applied.
            checkpoint_push_freq: Frequency to push checkpoints to Hub (in steps).
                                  If None, only pushes final model.
        """
        self.cfg = cfg
        self.sampler_config = sampler_config
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

        # Apply sampler config if provided
        if self.sampler_config is not None:
            cfg.dataset.episodes = self.sampler_config.episodes

        # Force pyav video decoding to avoid issues with torchcodec and GPU-accelerated video decoding in the cloud.
        cfg.dataset.video_backend = "pyav"

        # Check device is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.device("cuda")
        cfg.policy.device = "cuda"
        print(f"Using device: {device}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        logging.info("Creating dataset")
        dataset = make_dataset(
            cfg, plugins=[EpisodeOutcomePlugin(), PiStar0_6CumulativeRewardPlugin(normalise=True)]
        )

        # Create environment for evaluation
        eval_env = None
        if cfg.eval_freq > 0 and cfg.env is not None:
            logging.info("Creating env")
            eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

        logging.info("Creating policy")
        policy = make_rewact_policy(cfg.policy, dataset.meta)

        logging.info("Creating optimizer and scheduler")
        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
        grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

        step = 0

        # Try to resume from Hub checkpoint first
        if cfg.policy.push_to_hub and cfg.policy.repo_id:
            step = self._try_resume_from_hub(policy, optimizer, lr_scheduler, cfg)
        
        # Fall back to local checkpoint if specified
        if cfg.resume and step == 0:
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

        shuffle = True
        sampler = None

        import subprocess
        result = subprocess.run(['df', '-h', '/dev/shm'], capture_output=True, text=True)
        logging.info(f"Shared memory: {result.stdout}")

        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
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

            if is_saving_step:
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
            
            # Also save training state for resumption
            self._save_training_state_to_hub(optimizer, lr_scheduler, step, cfg.policy.repo_id)

        return {
            "model_repo_id": model_repo_id,
            "wandb_url": self.wandb_run_url,
            "final_loss": train_tracker.loss.avg,
            "total_steps": step,
        }

    def _save_training_state_to_hub(self, optimizer, lr_scheduler, step, repo_id):
        """Save training state (optimizer, scheduler, step) to Hub for resumption."""
        try:
            from huggingface_hub import HfApi
            import tempfile
            
            api = HfApi()
            
            # Save training state to temp file
            training_state = {
                "step": step,
                "optimizer": optimizer.state_dict(),
            }
            if lr_scheduler:
                training_state["lr_scheduler"] = lr_scheduler.state_dict()
            
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as f:
                torch.save(training_state, f)
                temp_path = f.name
            
            # Upload to Hub
            api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo="training_state.pth",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Save training state at step {step}",
            )
            
            # Clean up temp file
            Path(temp_path).unlink()
            logging.info(f"✓ Saved training state to Hub")
            
        except Exception as e:
            logging.warning(f"Failed to save training state to Hub: {e}")

    def _try_resume_from_hub(self, policy, optimizer, lr_scheduler, cfg) -> int:
        """Try to resume training from existing Hub checkpoint.
        
        Returns:
            step: The step number to resume from (0 if no checkpoint found)
        """
        try:
            from huggingface_hub import HfApi, hf_hub_download
            from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
            
            api = HfApi()
            repo_id = cfg.policy.repo_id
            
            logging.info(f"Checking for existing checkpoint on Hub: {repo_id}")
            
            # Check if repo exists
            try:
                repo_info = api.repo_info(repo_id, repo_type="model")
                logging.info(f"✓ Found existing model on Hub: {repo_id}")
            except (RepositoryNotFoundError, HfHubHTTPError):
                logging.info(f"No existing checkpoint found on Hub. Starting fresh training.")
                return 0
            
            # Try to download the checkpoint
            try:
                # Download pretrained_model.safetensors (the main model file)
                model_path = hf_hub_download(repo_id=repo_id, filename="pretrained_model.safetensors")
                
                # Load the model weights
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
                policy.load_state_dict(state_dict)
                logging.info(f"✓ Loaded model weights from Hub")
                
                # Try to get training state (optimizer, scheduler, step)
                try:
                    training_state_path = hf_hub_download(repo_id=repo_id, filename="training_state.pth")
                    training_state = torch.load(training_state_path, map_location="cpu")
                    
                    step = training_state.get("step", 0)
                    optimizer.load_state_dict(training_state.get("optimizer", optimizer.state_dict()))
                    if lr_scheduler and "lr_scheduler" in training_state:
                        lr_scheduler.load_state_dict(training_state["lr_scheduler"])
                    
                    logging.info(f"✓ Resuming training from step {step}")
                    return step
                except Exception as e:
                    # If training state doesn't exist, just use the model weights
                    logging.info(f"No training state found, starting from step 0 with pretrained weights")
                    return 0
                    
            except Exception as e:
                logging.warning(f"Could not load checkpoint from Hub: {e}")
                return 0
                
        except Exception as e:
            logging.warning(f"Error checking Hub for checkpoint: {e}")
            return 0

    def _push_checkpoint_to_hub(self, policy, cfg, step):
        """Push intermediate checkpoint to Hub."""
        try:            
            policy.push_model_to_hub(cfg)
            logging.info(f"Pushed checkpoint to {cfg.policy.repo_id}")
        except Exception as e:
            logging.warning(f"Failed to push checkpoint to Hub: {e}")
