"""
RewACT: Reward-Augmented Action Chunking with Transformers

A PyTorch implementation of RewACT, extending the ACT (Action Chunking with Transformers) 
model with reward-based learning for improved robotic control.
"""

__version__ = "0.1.0"
__author__ = "Ville Kuosmanen"

# Core components
from rewact.dataset_with_reward import LeRobotDatasetWithReward, KeypointReward
from rewact.reward_plugin import RewardPlugin, RewardPluginInstance
from rewact.policies.rewact import RewACTConfig, RewACTPolicy, RewACT
from rewact.policies.actvantage import ACTvantageConfig, ACTvantagePolicy, ACTvantage
from rewact.utils import WandBLogger, make_rewact_policy, make_actvantage_policy
from rewact.plugin_utils import get_plugin_instance

__all__ = [
    "LeRobotDatasetWithReward",
    "KeypointReward",
    "RewardPlugin",
    "RewardPluginInstance",
    "RewACTConfig", 
    "RewACTPolicy",
    "RewACT",
    "make_rewact_policy",
    "make_actvantage_policy",
    "get_plugin_instance",
    "WandBLogger",
    "ACTvantageConfig",
    "ACTvantagePolicy",
    "ACTvantage",
]
