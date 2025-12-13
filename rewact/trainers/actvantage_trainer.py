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
)
from robocandywrapper import WandBLogger
from robocandywrapper.plugins import EpisodeOutcomePlugin
from robocandywrapper import make_dataset

from rewact.plugins import PiStar0_6CumulativeRewardPlugin, ControlModePlugin, PiStar0_6AdvantagePlugin
from rewact.utils import make_actvantage_policy
from rewact.policies.factory import make_pre_post_processors


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

            step += 1
            train_tracker.step()
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
            is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
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
                if self.wandb_logger:
                    self.wandb_logger.log_policy(checkpoint_dir)

                # Push checkpoint to Hub if configured
                if is_checkpoint_push_step and cfg.policy.push_to_hub:
                    logging.info(f"Pushing checkpoint {step} to Hub")
                    self._push_checkpoint_to_hub(policy, cfg)

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
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)
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

    def _push_checkpoint_to_hub(self, policy, cfg):
        """Push intermediate checkpoint to Hub."""
        try:            
            policy.push_model_to_hub(cfg)
            logging.info(f"Pushed checkpoint to {cfg.policy.repo_id}")
        except Exception as e:
            logging.warning(f"Failed to push checkpoint to Hub: {e}")
