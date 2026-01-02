"""
Utility functions for working with the robocandywrapper plugin system.
"""

from typing import Optional, Type

from robocandywrapper import DatasetPlugin, PluginInstance, WrappedRobotDataset


def get_plugin_instance(
    wrapped_dataset: WrappedRobotDataset,
    plugin_type: Type[DatasetPlugin],
    dataset_idx: int = 0
) -> Optional[PluginInstance]:
    """
    Get a specific plugin instance from a wrapped dataset.
    
    This is a helper function that works around issues in the WrappedRobotDataset.get_plugin_instance method.
    
    Args:
        wrapped_dataset: The WrappedRobotDataset instance
        plugin_type: Type/class of the plugin to find
        dataset_idx: Index of the dataset (default: 0)
        
    Returns:
        The plugin instance, or None if not found
    """
    if dataset_idx >= len(wrapped_dataset._plugin_instances):
        return None
    
    # Find the plugin in the plugins list
    plugin_idx = None
    for i, plugin in enumerate(wrapped_dataset._plugins):
        if isinstance(plugin, plugin_type):
            plugin_idx = i
            break
    
    if plugin_idx is None:
        return None
    
    # Return the corresponding plugin instance
    return wrapped_dataset._plugin_instances[dataset_idx][plugin_idx]


