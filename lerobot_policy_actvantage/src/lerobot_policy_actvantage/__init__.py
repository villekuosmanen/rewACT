"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_actvantage import ACTvantageConfig
from .modeling_actvantage import ACTvantagePolicy, ACTvantage
from .processor_actvantage import make_actvantage_pre_post_processors

__all__ = [
    "ACTvantageConfig",
    "ACTvantagePolicy",
    "ACTvantage",
    "make_actvantage_pre_post_processors",
]


