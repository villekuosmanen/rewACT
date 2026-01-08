"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_rewact import RewACTConfig
from .modeling_rewact import RewACTPolicy, RewACT
from .processor_rewact import make_rewact_pre_post_processors

__all__ = [
    "RewACTConfig",
    "RewACTPolicy",
    "RewACT",
    "make_rewact_pre_post_processors",
]


