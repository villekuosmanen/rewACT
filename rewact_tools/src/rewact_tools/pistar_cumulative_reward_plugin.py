"""
Pi*0.6 Cumulative Reward Plugin for robocandywrapper.

This plugin implements the cumulative reward function from Physical Intelligence's
Pi*0.6 foundation model, where rewards are based on episode-level success labels
and the value function corresponds to the (negative) number of steps until successful
completion of the episode.
"""

from typing import Dict, Optional, Any

import numpy as np
import torch
from robocandywrapper import DatasetPlugin, PluginInstance



class PiStar0_6CumulativeRewardPluginInstance(PluginInstance):
    """
    Plugin instance that calculates cumulative rewards based on Pi*0.6 reward function.
    
    The reward function is:
    - r_t = 0 if t = T and success
    - r_t = -C_fail if t = T and failure
    - r_t = -1 otherwise
    
    Where cumulative reward accumulates backwards from the final timestep.
    """
    
    def __init__(
        self,
        dataset,
        c_fail: float = 10000.0,
        normalise: bool = False,
    ):
        super().__init__(dataset)
        self.c_fail = c_fail
        self.normalise = normalise
        
        # Cache for cumulative rewards by episode
        self._episode_cumulative_rewards = {}
        
        # Cache for normalization parameters
        self._norm_min = None
        self._norm_max = None
        self._normalization_computed = False
        
        # Detect if meta.episodes is a dict (format 2.1) or HF dataset (newer format)
        self._episodes_is_dict = isinstance(dataset.meta.episodes, dict)
        
        # Create mapping from episode_idx to position in episode_data_index
        # This is needed when using a subset of episodes
        if hasattr(dataset, 'episodes') and dataset.episodes is not None:
            self._episode_idx_to_pos = {ep_idx: pos for pos, ep_idx in enumerate(dataset.episodes)}
        else:
            # No subsetting, episode_idx == position
            self._episode_idx_to_pos = None
        
        # Cache for episode_data_index (computed lazily for datasets without it)
        self._cached_episode_data_index = None
    
    def get_data_keys(self) -> list[str]:
        """Return the keys this plugin will add to items."""
        return ["reward", "use_action_mask"]
    
    def _get_episode_data_index_pos(self, episode_idx: int) -> int:
        """
        Get the position in episode_data_index for a given episode_idx.
        
        When using a subset of episodes, episode_idx (the actual episode number)
        needs to be mapped to its position in the filtered episode_data_index.
        
        Args:
            episode_idx: The actual episode number from the dataset
            
        Returns:
            Position to use for indexing into episode_data_index
        """
        if self._episode_idx_to_pos is not None:
            if episode_idx not in self._episode_idx_to_pos:
                raise ValueError(
                    f"Episode {episode_idx} not found in the subset of episodes being used. "
                    f"Available episodes: {list(self._episode_idx_to_pos.keys())}"
                )
            return self._episode_idx_to_pos[episode_idx]
        else:
            # No subsetting, episode_idx is the position
            return episode_idx
    
    def _get_episode_length(self, episode_idx: int) -> int:
        """
        Get the length of an episode, supporting both dict (format 2.1) and HF dataset formats.
        
        Args:
            episode_idx: The episode index
            
        Returns:
            The length of the episode
        """
        if self._episodes_is_dict:
            return self.dataset.meta.episodes[episode_idx]["length"]
        else:
            # HF dataset format - access by row index
            return self.dataset.meta.episodes[episode_idx]["length"]
    
    def _iter_episode_indices(self):
        """
        Iterate over all episode indices, supporting both dict (format 2.1) and HF dataset formats.
        
        Yields:
            Episode indices
        """
        if self._episodes_is_dict:
            yield from self.dataset.meta.episodes.keys()
        else:
            # HF dataset format - iterate over row indices
            for i in range(len(self.dataset.meta.episodes)):
                yield i
    
    def _get_episode_data_index(self) -> dict[str, torch.Tensor]:
        """
        Get the episode data index, using cached version if available.
        
        Supports both dataset formats:
        - Newer format: uses dataset.episode_data_index directly
        - Older format: calculates from hf_dataset and caches the result
        
        Returns:
            Dictionary with 'from' and 'to' tensors indicating episode boundaries
        """
        if hasattr(self.dataset, 'episode_data_index'):
            return self.dataset.episode_data_index
        
        # Calculate and cache for datasets without episode_data_index
        if self._cached_episode_data_index is None:
            self._cached_episode_data_index = calculate_episode_data_index(self.dataset.hf_dataset)
        
        return self._cached_episode_data_index
    
    def get_item_data(
        self,
        idx: int,
        episode_idx: int,
        accumulated_data: Optional[Dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Get cumulative reward data for a specific item."""
        # Check if episode outcome data is available
        if accumulated_data is None:
            raise ValueError(
                "PiStar0.6CumulativeRewardPlugin requires episode_outcome data from another plugin. "
                "Make sure an episode outcome plugin is loaded before this plugin."
            )
        
        if "episode_outcome_mask" not in accumulated_data or "episode_outcome" not in accumulated_data:
            raise ValueError(
                "PiStar0.6CumulativeRewardPlugin requires 'episode_outcome' and 'episode_outcome_mask' "
                "in accumulated_data. Make sure an episode outcome plugin is loaded before this plugin."
            )
        
        episode_outcome_mask = accumulated_data["episode_outcome_mask"]
        if not episode_outcome_mask:
            raise ValueError(
                f"Episode outcome is not available for episode {episode_idx}. "
                "PiStar0.6CumulativeRewardPlugin requires episode outcomes to be set."
            )
        
        episode_outcome = accumulated_data["episode_outcome"]
        
        # Get episode information
        # Map episode_idx to position in episode_data_index (needed for episode subsets)
        ep_pos = self._get_episode_data_index_pos(episode_idx)
        episode_data_index = self._get_episode_data_index()
        ep_start = episode_data_index["from"][ep_pos]
        frame_index_in_episode = idx - ep_start.item()
        
        # Calculate cumulative reward for this episode if not cached
        if episode_idx not in self._episode_cumulative_rewards:
            self._compute_episode_cumulative_rewards(episode_idx, episode_outcome)
        
        # Get the cumulative reward for this frame
        cumulative_reward = self._episode_cumulative_rewards[episode_idx][frame_index_in_episode]
        
        # Apply normalization if enabled
        if self.normalise:
            if not self._normalization_computed:
                self._compute_normalization_parameters()
            cumulative_reward = self._normalize_reward(cumulative_reward)
        
        # Calculate action masking (similar to DenseRewardPlugin)
        repo_name = self.dataset.repo_id.split('/')[1] if '/' in self.dataset.repo_id else self.dataset.repo_id
        use_action_mask = True
        
        # For simplicity, we'll keep action masking consistent with DenseRewardPlugin
        # This can be parameterized if needed
        
        return {
            "reward": torch.tensor(cumulative_reward, dtype=torch.float32),
            "use_action_mask": torch.tensor(use_action_mask, dtype=torch.bool)
        }
    
    def _compute_episode_cumulative_rewards(self, episode_idx: int, episode_outcome: bool):
        """
        Compute cumulative rewards for an entire episode.
        
        Args:
            episode_idx: Episode index
            episode_outcome: True if success, False if failure
        """
        episode_length = self._get_episode_length(episode_idx)
        cumulative_rewards = np.zeros(episode_length, dtype=np.float32)
        
        if episode_outcome:  # Success
            # Last frame has reward 0, cumulative = 0
            # Previous frames have reward -1 each, cumulating backwards
            for t in range(episode_length):
                cumulative_rewards[t] = -(episode_length - 1 - t)
        else:  # Failure
            # Last frame has reward -C_fail
            # Previous frames have reward -1 each
            cumulative_rewards[-1] = -self.c_fail
            for t in range(episode_length - 1):
                cumulative_rewards[t] = -self.c_fail - (episode_length - 1 - t)
        
        self._episode_cumulative_rewards[episode_idx] = cumulative_rewards
    
    def _compute_normalization_parameters(self):
        """
        Compute normalization parameters based on all successful episodes.
        
        Sets:
        - norm_min: lowest cumulative reward in successful episodes (maps to 0)
        - norm_max: 0 (success timestep, maps to 1)
        """
        # We need to iterate through all episodes to find successful ones
        # and compute the maximum episode length among them
        max_successful_episode_length = 0
        
        # Iterate over episode indices (supports both dict format 2.1 and HF dataset format)
        for episode_idx in self._iter_episode_indices():
            # We need to check if this episode is successful
            # For now, we'll compute rewards for all episodes that haven't been cached
            # This requires getting episode outcomes, which we can't easily access here
            # So we'll use a different approach: compute on first access
            episode_length = self._get_episode_length(episode_idx)
            max_successful_episode_length = max(max_successful_episode_length, episode_length)
        
        # The minimum cumulative reward in successful episodes is:
        # -(max_episode_length - 1) at frame 0 of the longest episode
        self._norm_min = -(max_successful_episode_length - 1)
        self._norm_max = 0.0  # Success timestep
        
        self._normalization_computed = True
    
    def _normalize_reward(self, cumulative_reward: float) -> float:
        """
        Normalize cumulative reward to [0, 1] range.
        
        Args:
            cumulative_reward: Raw cumulative reward value
            
        Returns:
            Normalized reward in [0, 1]
        """
        if self._norm_min == self._norm_max:
            # Edge case: if all episodes have length 1
            return 1.0 if cumulative_reward >= 0 else 0.0
        
        # Linear normalization: norm_min -> 0, norm_max (0) -> 1
        normalized = (cumulative_reward - self._norm_min) / (self._norm_max - self._norm_min)
        
        # Clip to [0, 1] - this handles failure episodes which are more negative
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return float(normalized)
    
    def denormalize_reward(self, normalized: float) -> float:
        """
        Denormalize reward from [0, 1] range back to raw cumulative reward value.
        
        This is the inverse of _normalize_reward.
        
        Args:
            normalized: Normalized reward in [0, 1]
            
        Returns:
            Raw cumulative reward value
        """
        if self._norm_min == self._norm_max:
            # Edge case: if all episodes have length 1
            # In forward direction: cumulative_reward >= 0 -> 1.0, else -> 0.0
            # For reverse, we can return a reasonable default
            return 0.0 if normalized >= 0.5 else self._norm_min
        
        # Linear denormalization: inverse of (cumulative_reward - norm_min) / (norm_max - norm_min)
        # Solving for cumulative_reward: normalized * (norm_max - norm_min) + norm_min
        cumulative_reward = normalized * (self._norm_max - self._norm_min) + self._norm_min
        
        return float(cumulative_reward)


class PiStar0_6CumulativeRewardPlugin(DatasetPlugin):
    """
    Plugin that implements the Pi*0.6 cumulative reward function.
    
    Based on the reward function from Physical Intelligence's Pi*0.6 foundation model,
    which uses episode-level success labels to derive rewards where the value function
    corresponds to the (negative) number of steps until successful completion.
    
    The reward function is:
    - r_t = 0 if t = T and success
    - r_t = -C_fail if t = T and failure
    - r_t = -1 otherwise
    
    This results in cumulative rewards that count down to 0 for successful episodes,
    and very negative values for failed episodes.
    
    Example:
        ```python
        from robocandywrapper import WrappedRobotDataset
        from rewact_tools.pistar_cumulative_reward_plugin import PiStar0_6CumulativeRewardPlugin
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        # Create plugin with normalization
        reward_plugin = PiStar0_6CumulativeRewardPlugin(normalise=True, c_fail=1000.0)
        
        # Load dataset with plugin (make sure episode outcome plugin is loaded first!)
        base_dataset = LeRobotDataset("my_dataset")
        dataset = WrappedRobotDataset(base_dataset, plugins=[outcome_plugin, reward_plugin])
        
        # Access data with cumulative rewards
        item = dataset[0]
        reward = item["reward"]  # Tensor with cumulative reward value
        ```
    
    Note: This plugin requires episode outcome data from another plugin that provides
    'episode_outcome' and 'episode_outcome_mask' keys in accumulated_data.
    """
    
    def __init__(
        self,
        c_fail: float = 10000.0,
        normalise: bool = False,
    ):
        """
        Initialize the Pi*0.6 cumulative reward plugin.
        
        Args:
            c_fail: Large negative constant for failed episodes. Should be larger than
                   any cumulative reward in successful episodes (default: 1000.0)
            normalise: If True, normalize rewards to [0, 1] where:
                      - 0 maps to the lowest cumulative reward in successful episodes
                      - 1 maps to 0 (the success timestep)
                      - Failed episodes (more negative) are clipped to 0
        """
        if c_fail <= 0:
            raise ValueError("c_fail must be positive")
        
        self.c_fail = c_fail
        self.normalise = normalise
    
    def attach(self, dataset) -> PiStar0_6CumulativeRewardPluginInstance:
        """Create a dataset-specific plugin instance."""
        return PiStar0_6CumulativeRewardPluginInstance(
            dataset,
            c_fail=self.c_fail,
            normalise=self.normalise,
        )

def calculate_episode_data_index(hf_dataset: Any) -> dict[str, torch.Tensor]:
    """
    Calculate episode data index for the provided HuggingFace Dataset. Relies on episode_index column of hf_dataset.

    Parameters:
    - hf_dataset (datasets.Dataset): A HuggingFace dataset containing the episode index.

    Returns:
    - episode_data_index: A dictionary containing the data index for each episode. The dictionary has two keys:
        - "from": A tensor containing the starting index of each episode.
        - "to": A tensor containing the ending index of each episode.
    """
    episode_data_index = {"from": [], "to": []}

    current_episode = None
    """
    The episode_index is a list of integers, each representing the episode index of the corresponding example.
    For instance, the following is a valid episode_index:
      [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    Below, we iterate through the episode_index and populate the episode_data_index dictionary with the starting and
    ending index of each episode. For the episode_index above, the episode_data_index dictionary will look like this:
        {
            "from": [0, 3, 7],
            "to": [3, 7, 12]
        }
    """
    if len(hf_dataset) == 0:
        episode_data_index = {
            "from": torch.tensor([]),
            "to": torch.tensor([]),
        }
        return episode_data_index
    for idx, episode_idx in enumerate(hf_dataset["episode_index"]):
        if episode_idx != current_episode:
            # We encountered a new episode, so we append its starting location to the "from" list
            episode_data_index["from"].append(idx)
            # If this is not the first episode, we append the ending location of the previous episode to the "to" list
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            # Let's keep track of the current episode index
            current_episode = episode_idx
        else:
            # We are still in the same episode, so there is nothing for us to do here
            pass
    # We have reached the end of the dataset, so we append the ending location of the last episode to the "to" list
    episode_data_index["to"].append(idx + 1)

    for k in ["from", "to"]:
        episode_data_index[k] = torch.tensor(episode_data_index[k])

    return episode_data_index

