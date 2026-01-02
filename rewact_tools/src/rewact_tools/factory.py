from typing import Any
from typing_extensions import Unpack

from lerobot.processor import PolicyProcessorPipeline, PolicyAction, batch_to_transition, transition_to_batch
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.policies.factory import ProcessorConfigKwargs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

# Import from installed policy packages
try:
    from lerobot_policy_rewact import RewACTConfig, make_rewact_pre_post_processors
except ImportError:
    RewACTConfig = None
    make_rewact_pre_post_processors = None

try:
    from lerobot_policy_actvantage import ACTvantageConfig, make_actvantage_pre_post_processors
except ImportError:
    ACTvantageConfig = None
    make_actvantage_pre_post_processors = None

# Import utils from rewact_tools
from .utils import create_batch_to_transition

def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    plugin_features: dict[str, Any] | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    if pretrained_path:
        return (
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("preprocessor_overrides", {}),
                to_transition=create_batch_to_transition(plugin_features=plugin_features),
                to_output=transition_to_batch,
            ),
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("postprocessor_overrides", {}),
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            ),
        )

    # Create a new processor based on policy type
    # Here, only consider RewACT and ACTvantage processors - use LeRobot's factory methods for all other policies
    if RewACTConfig is not None and isinstance(policy_cfg, RewACTConfig):
        if make_rewact_pre_post_processors is None:
            raise ImportError("lerobot_policy_rewact is not installed. Please install it to use RewACT policies.")
        
        processors = make_rewact_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
            plugin_features=plugin_features,
        )
    elif ACTvantageConfig is not None and isinstance(policy_cfg, ACTvantageConfig):
        if make_actvantage_pre_post_processors is None:
            raise ImportError("lerobot_policy_actvantage is not installed. Please install it to use ACTvantage policies.")
        
        processors = make_actvantage_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
            plugin_features=plugin_features,
        )
    else:
        raise NotImplementedError(f"Processor for policy type '{policy_cfg.type}' is not implemented.")

    return processors

