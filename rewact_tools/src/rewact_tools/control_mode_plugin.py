"""
Control Mode Plugin for robocandywrapper.

This plugin loads control mode data from a DAgger data source JSON file
and provides control mode labels (human, policy, or unknown) during training.
"""

from typing import Dict, Optional, Any, List
import json
import torch
import warnings
from pathlib import Path
from robocandywrapper import DatasetPlugin, PluginInstance


# Control mode constants
CONTROL_MODE_POLICY = "policy"
CONTROL_MODE_HUMAN = "human"
CONTROL_MODE_UNKNOWN = "unknown"


class ControlModeSegment:
    """Represents a segment of frames with a specific control mode."""
    
    def __init__(self, start_index: int, end_index: int, mode: str):
        self.start_index = start_index
        self.end_index = end_index
        self.mode = mode
    
    def contains(self, frame_index: int) -> bool:
        """Check if a frame index falls within this segment (inclusive)."""
        return self.start_index <= frame_index <= self.end_index


class ControlModePluginInstance(PluginInstance):
    """
    Plugin instance that provides control mode labels for each frame.
    """
    
    def __init__(
        self,
        dataset,
        episode_modes: Dict[int, List[ControlModeSegment]],
    ):
        super().__init__(dataset)
        
        # Store episode modes data indexed by episode_idx (as int)
        self.episode_modes = episode_modes
        
        # Create mapping from episode_idx to position in episode_data_index
        # This is needed when using a subset of episodes
        if hasattr(dataset, 'episodes') and dataset.episodes is not None:
            self._episode_idx_to_pos = {ep_idx: pos for pos, ep_idx in enumerate(dataset.episodes)}
        else:
            # No subsetting, episode_idx == position
            self._episode_idx_to_pos = None
        
        # Cache for episode_data_index (computed lazily for datasets without it)
        self._cached_episode_data_index = None
        
        # Log statistics
        total_episodes = len(self.episode_modes)
        episodes_with_data = sum(1 for segments in self.episode_modes.values() if segments)
        print(f"Control Mode plugin initialized:")
        print(f"  Total episodes with mode data: {episodes_with_data}/{total_episodes}")
    
    def get_data_keys(self) -> list[str]:
        """Return the keys this plugin will add to items."""
        return ["control_mode", "control_mode_autonomous"]
    
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
        """Get control mode data for a specific item."""
        
        # Get episode information to determine frame index within episode
        ep_pos = self._get_episode_data_index_pos(episode_idx)
        episode_data_index = self._get_episode_data_index()
        ep_start = episode_data_index["from"][ep_pos]
        frame_index_in_episode = idx - ep_start.item()
        
        # Look up control mode for this episode and frame
        control_mode = self._get_control_mode(episode_idx, frame_index_in_episode)
        
        return {
            "control_mode": control_mode,
            "control_mode_autonomous": torch.tensor([control_mode == CONTROL_MODE_POLICY], dtype=torch.bool)
        }
    
    def _get_control_mode(self, episode_idx: int, frame_index_in_episode: int) -> str:
        """
        Get the control mode for a specific frame in an episode.
        
        Args:
            episode_idx: Episode index
            frame_index_in_episode: Frame index within the episode (0-based)
            
        Returns:
            Control mode string (CONTROL_MODE_POLICY, CONTROL_MODE_HUMAN, or CONTROL_MODE_UNKNOWN)
        """
        # Check if episode has mode data
        if episode_idx not in self.episode_modes:
            return CONTROL_MODE_UNKNOWN
        
        segments = self.episode_modes[episode_idx]
        if not segments:
            return CONTROL_MODE_UNKNOWN
        
        # Find the segment that contains this frame
        for segment in segments:
            if segment.contains(frame_index_in_episode):
                return segment.mode
        
        # Frame not found in any segment
        return CONTROL_MODE_UNKNOWN


class ControlModePlugin(DatasetPlugin):
    """
    Plugin that provides control mode labels from DAgger data collection.
    
    Control modes are loaded from a JSON file in the dataset repository at
    `dagger_data_source/episode_modes.json`. The file should have the following structure:
    
    ```json
    {
        "0": [
            {"start_index": 0, "end_index": 57, "mode": "policy"},
            {"start_index": 58, "end_index": 861, "mode": "unknown"}
        ],
        "1": [
            {"start_index": 0, "end_index": 85, "mode": "policy"}
        ]
    }
    ```
    
    Where episode IDs are strings, and each episode contains segments with:
    - start_index: Starting frame index (inclusive)
    - end_index: Ending frame index (inclusive)
    - mode: Control mode ("policy", "human", or "unknown")
    
    If the file is missing, episodes are not present, or frames are not covered,
    the control mode will default to "unknown".
    
    Example:
    ```python
    from robocandywrapper import WrappedRobotDataset
    from rewact_tools.control_mode_plugin import ControlModePlugin
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    # Create plugin
    control_mode_plugin = ControlModePlugin()
    
    # Load dataset with plugin
    base_dataset = LeRobotDataset("my_dataset")
    dataset = WrappedRobotDataset(base_dataset, plugins=[control_mode_plugin])
    
    # Access data with control modes
    item = dataset[0]
    control_mode = item["control_mode"]  # String: "policy", "human", or "unknown"
    ```
    """
    
    def __init__(
        self,
        episode_modes_file: Optional[str | Path] = None,
    ):
        """
        Initialize the control mode plugin.
        
        Args:
            episode_modes_file: Path to the episode modes JSON file. If None, the plugin
                              will attempt to find it at 'dagger_data_source/episode_modes.json'
                              relative to the dataset root when attached.
        """
        self.episode_modes_file = Path(episode_modes_file) if episode_modes_file else None
        self.episode_modes_data = None
    
    def attach(self, dataset) -> ControlModePluginInstance:
        """Create a dataset-specific plugin instance."""
        
        # Determine the path to the episode modes file
        if self.episode_modes_file is None:
            # Try to find it in the dataset root
            if hasattr(dataset, 'root'):
                dataset_root = Path(dataset.root)
            elif hasattr(dataset, 'local_dir'):
                dataset_root = Path(dataset.local_dir)
            else:
                warnings.warn(
                    "Could not determine dataset root directory. "
                    "Control modes will default to 'unknown'. "
                    "Please provide episode_modes_file explicitly."
                )
                episode_modes_file = None
            
            if dataset_root:
                episode_modes_file = dataset_root / "dagger_data_source" / "episode_modes.json"
        else:
            episode_modes_file = self.episode_modes_file
        
        # Load episode modes data
        episode_modes = self._load_episode_modes(episode_modes_file)
        
        return ControlModePluginInstance(
            dataset,
            episode_modes=episode_modes,
        )
    
    def _load_episode_modes(self, file_path: Optional[Path]) -> Dict[int, List[ControlModeSegment]]:
        """
        Load episode modes from JSON file.
        
        Args:
            file_path: Path to the episode modes JSON file
            
        Returns:
            Dictionary mapping episode_idx (int) to list of ControlModeSegment objects
        """
        if file_path is None or not file_path.exists():
            if file_path is not None:
                warnings.warn(
                    f"Episode modes file not found: {file_path}. "
                    "All control modes will default to 'unknown'."
                )
            return {}
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert to internal format
            episode_modes = {}
            for episode_id_str, segments_data in data.items():
                episode_idx = int(episode_id_str)
                segments = []
                
                for segment_data in segments_data:
                    segment = ControlModeSegment(
                        start_index=segment_data["start_index"],
                        end_index=segment_data["end_index"],
                        mode=segment_data["mode"]
                    )
                    segments.append(segment)
                
                episode_modes[episode_idx] = segments
            
            return episode_modes
            
        except Exception as e:
            warnings.warn(
                f"Error loading episode modes file {file_path}: {e}. "
                "All control modes will default to 'unknown'."
            )
            return {}

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
