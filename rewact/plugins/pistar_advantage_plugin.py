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
        self.advantage_data = advantage_data.set_index('frame_index')['advantage']
        
        # Threshold for binarizing advantage
        if use_percentile_threshold:
            # Use percentile of the data
            self.advantage_threshold = self.advantage_data.quantile(percentile / 100.0)
        elif advantage_threshold is not None:
            self.advantage_threshold = advantage_threshold
        else:
            # Default: use 0 (neutral)
            self.advantage_threshold = 0.0
                
        print(f"Advantage plugin initialized:")
        print(f"  Threshold: {self.advantage_threshold:.4f}")
        print(f"  Positive advantages: {(self.advantage_data > self.advantage_threshold).mean()*100:.1f}%")
    
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
        
        # Get pre-computed advantage value
        if idx in self.advantage_data.index:
            advantage_value = self.advantage_data.loc[idx]
        else:
            # Fallback: neutral advantage if not found
            advantage_value = 0.0
        
        return {
            "advantage": torch.tensor([1 if advantage_value > self.advantage_threshold else -1], dtype=torch.float32)  # (1, 1)
        }


class PiStar0_6AdvantagePlugin(DatasetPlugin):
    """
    Plugin that provides pre-computed advantage values for advantage-conditioned training.
    
    Advantages should be pre-computed using the value function and stored in a parquet file
    with columns: ['frame_index', 'advantage'].
    
    Example:
```python
        from robocandywrapper import WrappedRobotDataset
        from rewact.advantage_plugin import AdvantagePlugin
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        # Create plugin
        advantage_plugin = AdvantagePlugin(
            advantage_file="advantages.parquet",
            use_percentile_threshold=True,
            percentile=30.0  # Top 70% are positive
        )
        
        # Load dataset with plugin
        base_dataset = LeRobotDataset("my_dataset")
        dataset = WrappedRobotDataset(base_dataset, plugins=[advantage_plugin])
        
        # Access data with advantages
        item = dataset[0]
        advantage = item["advantage"]  # Tensor (1, 1) with advantage value
```
    """
    
    def __init__(
        self,
        advantage_file: str | Path,
        advantage_threshold: Optional[float] = None,
        use_percentile_threshold: bool = True,
        percentile: float = 30.0,
    ):
        """
        Initialize the advantage plugin.
        
        Args:
            advantage_file: Path to parquet file with columns ['frame_index', 'advantage']
            advantage_threshold: Fixed threshold for binarizing advantages (if not using percentile)
            use_percentile_threshold: If True, use percentile of data as threshold
            percentile: Percentile to use as threshold (e.g., 30 means top 70% are positive)
        """
        self.advantage_file = Path(advantage_file)
        if not self.advantage_file.exists():
            raise FileNotFoundError(f"Advantage file not found: {advantage_file}")
        
        # Load advantage data
        self.advantage_data = pd.read_parquet(self.advantage_file)
        
        if 'frame_index' not in self.advantage_data.columns or 'advantage' not in self.advantage_data.columns:
            raise ValueError("Advantage file must contain 'frame_index' and 'advantage' columns")
        
        self.advantage_threshold = advantage_threshold
        self.use_percentile_threshold = use_percentile_threshold
        self.percentile = percentile
    
    def attach(self, dataset) -> PiStar0_6AdvantagePluginInstance:
        """Create a dataset-specific plugin instance."""
        return PiStar0_6AdvantagePluginInstance(
            dataset,
            advantage_data=self.advantage_data,
            advantage_threshold=self.advantage_threshold,
            use_percentile_threshold=self.use_percentile_threshold,
            percentile=self.percentile,
        )
