"""
Utility functions for RewACT package.
"""
import logging
import os
import re
from pathlib import Path
from typing import Optional

from glob import glob
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from lerobot.constants import PRETRAINED_MODEL_DIR
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from termcolor import colored

from rewact.policy import RewACTPolicy

def cfg_to_group(cfg: TrainPipelineConfig, return_list: bool = False) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""
    dataset_tag = cfg.dataset.repo_id
    if dataset_tag.startswith('['):
        tags = dataset_tag.strip('[]').split(',')
        dataset_tag = f"{tags[0].strip()}_and_more"
    lst = [
        f"policy:{cfg.policy.type}",
        f"dataset:{dataset_tag}",
        f"seed:{cfg.seed}",
    ]
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    return lst if return_list else "-".join(lst)

def make_rewact_policy(
    cfg: PreTrainedConfig,
    ds_meta: Optional[LeRobotDatasetMetadata] = None,
    device: Optional[str] = None,
) -> RewACTPolicy:
    """
    Create a RewACT policy from configuration and dataset metadata.
    
    This function encapsulates the common policy creation logic used across
    training and visualization scripts.
    
    Args:
        cfg: Policy configuration (should be RewACTConfig or compatible)
        ds_meta: Dataset metadata containing features and stats
        device: Device to move the policy to. If None, uses cfg.device
        
    Returns:
        Initialized RewACTPolicy instance
        
    Raises:
        ValueError: If required metadata is missing
    """
    if ds_meta is None:
        raise ValueError("Dataset metadata (ds_meta) is required for policy creation")
    
    # Create policy kwargs
    kwargs = {}
    
    # Convert dataset features to policy features
    features = dataset_to_policy_features(ds_meta.features)
    kwargs["dataset_stats"] = ds_meta.stats
    
    # Set input and output features
    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg
    
    # Create policy instance
    if getattr(cfg, 'pretrained_path', None):
        # Load a pretrained policy and override the config if needed
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = RewACTPolicy.from_pretrained(**kwargs)
    else:
        # Make a fresh policy
        policy = RewACTPolicy(**kwargs)
    
    # Move to device
    target_device = device if device is not None else getattr(cfg, 'device', 'cpu')
    policy.to(target_device)
    
    return policy

def get_safe_wandb_artifact_name(name: str):
    """WandB artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")

def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    # Get the WandB run ID.
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    wandb_run_id = match.groups(0)[0]
    return wandb_run_id

class WandBLogger:
    """A helper class to log object using wandb."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        # Set up WandB.
        os.environ["WANDB_SILENT"] = "True"
        import wandb

        resume = "must" if cfg.resume else None
        try:
            wandb_run_id = (
                cfg.wandb.run_id
                if cfg.wandb.run_id
                else get_wandb_run_id_from_filesystem(self.log_dir)
                if cfg.resume
                else None
            )
        except RuntimeError:
            wandb_run_id = None
            resume = None
        wandb.init(
            id=wandb_run_id,
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.job_name,
            notes=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True),
            dir=self.log_dir,
            config=cfg.to_dict(),
            # TODO(rcadene): try set to True
            save_code=False,
            # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
            job_type="train_eval",
            resume=resume,
            mode=self.cfg.mode if self.cfg.mode in ["online", "offline", "disabled"] else "online",
        )
        print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")
        self._wandb = wandb

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to wandb."""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}_{step_id}"
        artifact_name = get_safe_wandb_artifact_name(artifact_name)
        artifact = self._wandb.Artifact(artifact_name, type="model")
        artifact.add_file(checkpoint_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE)
        self._wandb.log_artifact(artifact)

    def log_dict(self, d: dict, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        for k, v in d.items():
            if not isinstance(v, (int, float, str)):
                logging.warning(
                    f'WandB logging of key "{k}" was ignored as its type is not handled by this wrapper.'
                )
                continue
            self._wandb.log({f"{mode}/{k}": v}, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        wandb_video = self._wandb.Video(video_path, fps=self.env_fps, format="mp4")
        self._wandb.log({f"{mode}/video": wandb_video}, step=step)
