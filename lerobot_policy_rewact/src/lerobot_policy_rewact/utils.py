from typing import Any, Callable
from lerobot.processor.core import EnvTransition
from lerobot.processor import PolicyAction
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD, DONE, TRUNCATED


def create_batch_to_transition(plugin_features: dict[str, Any]) -> Callable[[dict[str, Any]], EnvTransition]:
    return lambda batch: batch_to_transition(batch, plugin_features)

def batch_to_transition(batch: dict[str, Any], plugin_features: dict[str, Any]) -> EnvTransition:
    # Validate input type.
    if not isinstance(batch, dict):
        raise ValueError(f"EnvTransition must be a dictionary. Got {type(batch).__name__}")

    action = batch.get(ACTION)
    if action is not None and not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

    # Extract observation and complementary data keys.
    observation_keys = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
    complementary_data = _extract_complementary_data(batch, plugin_features)

    return create_transition(
        observation=observation_keys if observation_keys else None,
        action=batch.get(ACTION),
        reward=batch.get(REWARD, 0.0),
        done=batch.get(DONE, False),
        truncated=batch.get(TRUNCATED, False),
        info=batch.get("info", {}),
        complementary_data=complementary_data if complementary_data else None,
    )

def _extract_complementary_data(batch: dict[str, Any], plugin_features: dict[str, Any]) -> dict[str, Any]:
    """
    Extract complementary data from a batch dictionary.

    This includes padding flags, task description, and indices.

    Args:
        batch: The batch dictionary.

    Returns:
        A dictionary with the extracted complementary data.
    """
    pad_keys = {k: v for k, v in batch.items() if "_is_pad" in k}
    task_key = {"task": batch["task"]} if "task" in batch else {}
    index_key = {"index": batch["index"]} if "index" in batch else {}
    task_index_key = {"task_index": batch["task_index"]} if "task_index" in batch else {}
    plugin_features_key = {k: batch[k] for k in plugin_features.keys() if k in batch}

    complementary_data = {**pad_keys, **task_key, **index_key, **task_index_key, **plugin_features_key}
    return complementary_data if complementary_data else None

