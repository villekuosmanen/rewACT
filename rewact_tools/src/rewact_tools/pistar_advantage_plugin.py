"""
Advantage Plugin for robocandywrapper.

This plugin loads pre-computed advantage values from a parquet file
and provides them during training for advantage-conditioned policies.
"""

from typing import Dict, Optional, Any
import torch
import pandas as pd
from pathlib import Path
from robocandywrapper import DatasetPlugin, PluginInstance
from .control_mode_plugin import CONTROL_MODE_HUMAN


class PiStar0_6AdvantagePluginInstance(PluginInstance):
    """
    Plugin instance that provides pre-computed advantage values.
    """
    
    def __init__(
        self,
        dataset,
        advantage_data: pd.DataFrame,
        advantage_threshold: Optional[float] = None,
        use_percentile_threshold: bool = True,
        percentile: float = 30.0,
    ):
        super().__init__(dataset)
        
        # Store advantage data indexed by frame_index
        indexed_data = advantage_data.set_index('frame_index')
        self.advantage_data = indexed_data['advantage']
        
        # Store required_intervention data if available
        if 'required_intervention' in indexed_data.columns:
            self.required_intervention_data = indexed_data['required_intervention']
            has_interventions = self.required_intervention_data.any()
            intervention_pct = self.required_intervention_data.mean() * 100
        else:
            self.required_intervention_data = None
            has_interventions = False
            intervention_pct = 0.0
        
        # Threshold for binarizing advantage
        if use_percentile_threshold:
            # Use percentile of this dataset's data (single dataset mode)
            self.advantage_threshold = self.advantage_data.quantile(percentile / 100.0)
            threshold_source = f"{percentile}th percentile (per-dataset)"
        elif advantage_threshold is not None:
            # Use provided threshold (multi-dataset mode with global threshold)
            self.advantage_threshold = advantage_threshold
            threshold_source = "global (across all datasets)"
        else:
            # Default: use 0 (neutral)
            self.advantage_threshold = 0.0
            threshold_source = "default (0.0)"
                
        print(f"  Dataset: {dataset.repo_id}")
        print(f"    Threshold: {self.advantage_threshold:.4f} ({threshold_source})")
        print(f"    Positive advantages in this dataset: {(self.advantage_data > self.advantage_threshold).mean()*100:.1f}%")
        if has_interventions:
            print(f"    Required intervention frames: {intervention_pct:.1f}%")
    
    def get_data_keys(self) -> list[str]:
        """Return the keys this plugin will add to items."""
        return ["advantage"]
    
    def get_item_data(
        self,
        idx: int,
        episode_idx: int,
        accumulated_data: Optional[Dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Get advantage data for a specific item."""
        
        # Check if control mode indicates human control
        # Human-controlled frames always get positive advantage (1)
        if accumulated_data is not None and "control_mode" in accumulated_data:
            control_mode = accumulated_data["control_mode"]
            if control_mode == CONTROL_MODE_HUMAN:
                return {
                    "advantage": torch.tensor([1], dtype=torch.float32)
                }
        
        # Check if this frame required intervention in the future
        # If so, it gets negative advantage (-1) regardless of computed advantage
        if self.required_intervention_data is not None and idx in self.required_intervention_data.index:
            required_intervention = self.required_intervention_data.loc[idx]
            if required_intervention:
                return {
                    "advantage": torch.tensor([-1], dtype=torch.float32)
                }
        
        # For policy and unknown modes, use pre-computed advantage values
        # Get pre-computed advantage value
        if idx in self.advantage_data.index:
            advantage_value = self.advantage_data.loc[idx]
        else:
            # Fallback: neutral advantage if not found
            advantage_value = 0.0
        
        return {
            "advantage": torch.tensor([1 if advantage_value > self.advantage_threshold else -1], dtype=torch.float32)
        }


class PiStar0_6AdvantagePlugin(DatasetPlugin):
    """
    Plugin that provides pre-computed advantage values for advantage-conditioned training.
    
    Advantages should be pre-computed using the value function and stored in a parquet file
    with columns: ['frame_index', 'advantage', 'required_intervention' (optional)].
    
    Advantage Assignment Logic (in priority order):
    1. Human-controlled frames (from ControlModePlugin): Always get advantage = 1
    2. Frames requiring future intervention (required_intervention=True): Always get advantage = -1
    3. Otherwise: Use threshold comparison on computed advantage value
    
    Integration with ControlModePlugin:
    If control mode data is available from ControlModePlugin, frames labeled as human-controlled
    will always receive a positive advantage value (1), regardless of the pre-computed advantage.
    This ensures that human demonstrations are always treated as high-quality examples.
    
    Required Intervention Detection:
    The 'required_intervention' column tracks frames where the N-step-ahead future required human
    intervention (current frame is policy-controlled but future frame is human-controlled).
    These frames get negative advantage (-1) as they represent actions leading to intervention.
    
    Example (single dataset):
```python
        from robocandywrapper import WrappedRobotDataset
        from rewact_tools import PiStar0_6AdvantagePlugin, ControlModePlugin
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        # Create plugins
        control_mode_plugin = ControlModePlugin()  # Optional: provides human/policy labels
        advantage_plugin = PiStar0_6AdvantagePlugin(
            advantage_file="advantages_dir",  # Can be a directory or single parquet file
            use_percentile_threshold=True,
            percentile=30.0  # Top 70% are positive
        )
        
        # Load dataset with plugins (ControlModePlugin first for advantage override)
        base_dataset = LeRobotDataset("my_dataset")
        dataset = WrappedRobotDataset(
            base_dataset, 
            plugins=[control_mode_plugin, advantage_plugin]
        )
        
        # Access data with advantages
        item = dataset[0]
        advantage = item["advantage"]  # Tensor with value 1 or -1
        # If control_mode is "human", advantage will be 1
        # Otherwise, based on pre-computed advantages
```
    
    Example (multiple datasets):
```python
        # For multiple datasets, provide a mapping of repo_id to advantage directory
        advantage_plugin = PiStar0_6AdvantagePlugin(
            advantage_file={
                "lerobot/dataset1": "outputs/dataset1_advantages",
                "lerobot/dataset2": "outputs/dataset2_advantages",
            },
            use_percentile_threshold=True,
            percentile=30.0
        )
```
    """
    
    def __init__(
        self,
        advantage_file: str | Path | dict[str, str | Path],
        advantage_threshold: Optional[float] = None,
        use_percentile_threshold: bool = True,
        percentile: float = 30.0,
    ):
        """
        Initialize the advantage plugin.
        
        Args:
            advantage_file: Path to parquet file OR directory containing episode_*.parquet files
                           OR dict mapping dataset repo_ids to their advantage directories
            advantage_threshold: Fixed threshold for binarizing advantages (if not using percentile)
            use_percentile_threshold: If True, use percentile of data as threshold
            percentile: Percentile to use as threshold (e.g., 30 means top 70% are positive)
        """
        # Store configuration
        self.advantage_file = advantage_file
        self.use_percentile_threshold = use_percentile_threshold
        self.percentile = percentile
        
        # Load advantage data and compute global threshold
        self.advantage_data = None  # For single dataset case
        self.advantage_data_dict = {}  # For multi-dataset case
        
        if isinstance(advantage_file, dict):
            # Multiple datasets: load all and compute global threshold
            print(f"Loading advantages from {len(advantage_file)} datasets for global threshold computation...")
            all_advantages = []
            
            for repo_id, adv_path in advantage_file.items():
                data = self._load_advantage_data(adv_path)
                self.advantage_data_dict[repo_id] = data
                all_advantages.extend(data['advantage'].values)
                print(f"  - {repo_id}: {len(data)} frames")
            
            # Compute global threshold across all datasets
            all_advantages_series = pd.Series(all_advantages)
            if use_percentile_threshold:
                self.advantage_threshold = all_advantages_series.quantile(percentile / 100.0)
            elif advantage_threshold is not None:
                self.advantage_threshold = advantage_threshold
            else:
                self.advantage_threshold = 0.0
            
            print(f"\nGlobal advantage threshold: {self.advantage_threshold:.4f}")
            print(f"Total frames across all datasets: {len(all_advantages)}")
            print(f"Positive advantages (global): {(all_advantages_series > self.advantage_threshold).mean()*100:.1f}%")
            
        else:
            # Single dataset: load data immediately (backward compatibility)
            self.advantage_data = self._load_advantage_data(advantage_file)
            
            # Compute threshold for single dataset
            if use_percentile_threshold:
                self.advantage_threshold = self.advantage_data['advantage'].quantile(percentile / 100.0)
            elif advantage_threshold is not None:
                self.advantage_threshold = advantage_threshold
            else:
                self.advantage_threshold = 0.0
    
    def _load_advantage_data(self, advantage_path: str | Path) -> pd.DataFrame:
        """Load advantage data from a file or directory."""
        advantage_path = Path(advantage_path)
        if not advantage_path.exists():
            raise FileNotFoundError(f"Advantage path not found: {advantage_path}")
        
        # Load advantage data from file or directory
        if advantage_path.is_dir():
            # Load all episode files from directory
            episode_files = sorted(advantage_path.glob("episode_*.parquet"))
            if not episode_files:
                raise FileNotFoundError(f"No episode_*.parquet files found in directory: {advantage_path}")
            
            print(f"Loading {len(episode_files)} episode advantage files from {advantage_path}")
            dfs = []
            for ep_file in episode_files:
                dfs.append(pd.read_parquet(ep_file))
            advantage_data = pd.concat(dfs, ignore_index=True)
        else:
            # Load single parquet file (backward compatibility)
            advantage_data = pd.read_parquet(advantage_path)
        
        if 'frame_index' not in advantage_data.columns or 'advantage' not in advantage_data.columns:
            raise ValueError("Advantage file must contain 'frame_index' and 'advantage' columns")
        
        # Check if required_intervention column exists (optional for backward compatibility)
        if 'required_intervention' not in advantage_data.columns:
            print(f"  Note: 'required_intervention' column not found in {advantage_path}, will use only advantage values")
        
        return advantage_data
    
    def attach(self, dataset) -> PiStar0_6AdvantagePluginInstance:
        """Create a dataset-specific plugin instance."""
        # Determine which advantage data to use
        if isinstance(self.advantage_file, dict):
            # Multiple datasets: use pre-loaded data and global threshold
            repo_id = dataset.repo_id
            if repo_id not in self.advantage_data_dict:
                raise KeyError(
                    f"No advantage data loaded for dataset '{repo_id}'. "
                    f"Available datasets: {list(self.advantage_data_dict.keys())}"
                )
            
            print(f"Attaching advantage plugin to dataset: {repo_id}")
            advantage_data = self.advantage_data_dict[repo_id]
        else:
            # Single dataset: use pre-loaded data
            advantage_data = self.advantage_data
        
        # Use the global threshold (already computed in __init__)
        return PiStar0_6AdvantagePluginInstance(
            dataset,
            advantage_data=advantage_data,
            advantage_threshold=self.advantage_threshold,
            use_percentile_threshold=False,  # Don't recalculate, use pre-computed threshold
            percentile=self.percentile,
        )
