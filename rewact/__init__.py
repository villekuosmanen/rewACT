"""
RewACT: Reward-Augmented Action Chunking with Transformers

A PyTorch implementation of RewACT, extending the ACT (Action Chunking with Transformers) 
model with reward-based learning for improved robotic control.
"""

__version__ = "0.1.0"
__author__ = "Ville Kuosmanen"

# Core components
from rewact.dataset_with_reward import LeRobotDatasetWithReward
from rewact.policy import RewACTConfig, RewACTPolicy, RewACT
from rewact.utils import make_rewact_policy

__all__ = [
    "LeRobotDatasetWithReward",
    "RewACTConfig", 
    "RewACTPolicy",
    "RewACT",
    "make_rewact_policy",
]
