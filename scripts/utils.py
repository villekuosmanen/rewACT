"""
Utility functions for training and visualization scripts.
"""
from typing import Optional
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType

# Import from installed policy packages
try:
    from lerobot_policy_rewact import RewACTPolicy, RewACTConfig
except ImportError:
    RewACTPolicy = None
    RewACTConfig = None

try:
    from lerobot_policy_actvantage import ACTvantagePolicy, ACTvantageConfig
except ImportError:
    ACTvantagePolicy = None
    ACTvantageConfig = None


def make_rewact_policy(
    cfg: PreTrainedConfig,
    ds_meta: Optional[LeRobotDatasetMetadata] = None,
    device: Optional[str] = None,
):
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
        ImportError: If lerobot_policy_rewact is not installed
    """
    if RewACTPolicy is None:
        raise ImportError("lerobot_policy_rewact is not installed. Please install it to use RewACT policies.")
    
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


def make_actvantage_policy(
    cfg: PreTrainedConfig,
    ds_meta: Optional[LeRobotDatasetMetadata] = None,
    device: Optional[str] = None,
):
    """
    Create an ACTvantage policy from configuration and dataset metadata.
    
    Raises:
        ImportError: If lerobot_policy_actvantage is not installed
    """
    if ACTvantagePolicy is None:
        raise ImportError("lerobot_policy_actvantage is not installed. Please install it to use ACTvantage policies.")
    
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
        policy = ACTvantagePolicy.from_pretrained(**kwargs)
    else:
        # Make a fresh policy
        policy = ACTvantagePolicy(**kwargs)
    
    # Move to device
    target_device = device if device is not None else getattr(cfg, 'device', 'cpu')
    policy.to(target_device)
    
    return policy


