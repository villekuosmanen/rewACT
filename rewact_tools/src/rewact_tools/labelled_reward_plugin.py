"""
Labelled Reward Plugin for robocandywrapper.

This plugin loads pre-computed per-frame reward labels from a parquet file
in the candywrapper_plugins directory. Rewards are produced offline by a
labelling pipeline (e.g. TOPReward) and stored per-episode so that they
can be used as training targets.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from robocandywrapper import CANDYWRAPPER_PLUGINS_DIR, DatasetPlugin, PluginInstance


LABELLED_REWARD_SUBDIR = "labelled_rewards"


class LabelledRewardPluginInstance(PluginInstance):
    """
    Plugin instance that serves pre-computed reward labels for each frame.
    """

    def __init__(self, dataset, reward_lookup: Dict[int, float]):
        super().__init__(dataset)
        self.reward_lookup = reward_lookup

        if hasattr(dataset, "episodes") and dataset.episodes is not None:
            self._episode_idx_to_pos = {
                ep_idx: pos for pos, ep_idx in enumerate(dataset.episodes)
            }
        else:
            self._episode_idx_to_pos = None

    def get_data_keys(self) -> list[str]:
        return ["reward", "use_action_mask"]

    def _get_episode_data_index_pos(self, episode_idx: int) -> int:
        if self._episode_idx_to_pos is not None:
            if episode_idx not in self._episode_idx_to_pos:
                raise ValueError(
                    f"Episode {episode_idx} not found in the subset of episodes. "
                    f"Available: {list(self._episode_idx_to_pos.keys())}"
                )
            return self._episode_idx_to_pos[episode_idx]
        return episode_idx

    def get_item_data(
        self,
        idx: int,
        episode_idx: int,
        accumulated_data: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if idx in self.reward_lookup:
            reward = self.reward_lookup[idx]
        else:
            warnings.warn(
                f"No labelled reward for global index {idx} "
                f"(episode {episode_idx}). Defaulting to 0.0."
            )
            reward = 0.0

        return {
            "reward": torch.tensor(reward, dtype=torch.float32),
            "use_action_mask": torch.tensor(True, dtype=torch.bool),
        }


class LabelledRewardPlugin(DatasetPlugin):
    """
    Plugin that loads pre-computed per-frame reward labels.

    Labels are stored as parquet files inside the dataset's
    ``candywrapper_plugins/labelled_rewards/`` directory, with one file per
    episode (``episode_000000.parquet``, ``episode_000001.parquet``, ...).

    Each parquet file must contain at least:
      - ``frame_index`` (int): global frame index in the dataset
      - ``reward`` (float): reward value (typically in [0, 1])

    Example::

        from robocandywrapper import WrappedRobotDataset
        from rewact_tools.labelled_reward_plugin import LabelledRewardPlugin
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        plugin = LabelledRewardPlugin()
        base = LeRobotDataset("my_dataset")
        dataset = WrappedRobotDataset(base, plugins=[plugin])
        item = dataset[0]
        reward = item["reward"]
    """

    def __init__(self, rewards_dir: Optional[str | Path] = None):
        """
        Args:
            rewards_dir: Explicit path to the directory containing
                episode parquet files. When ``None`` (the default), the
                plugin discovers the directory automatically from the
                dataset root at attach time.
        """
        self.rewards_dir = Path(rewards_dir) if rewards_dir is not None else None

    def attach(self, dataset) -> LabelledRewardPluginInstance:
        if self.rewards_dir is not None:
            rewards_dir = self.rewards_dir
        else:
            if hasattr(dataset, "root"):
                dataset_root = Path(dataset.root)
            elif hasattr(dataset, "local_dir"):
                dataset_root = Path(dataset.local_dir)
            else:
                raise RuntimeError(
                    "Cannot determine dataset root. "
                    "Please provide rewards_dir explicitly."
                )
            rewards_dir = (
                dataset_root / CANDYWRAPPER_PLUGINS_DIR / LABELLED_REWARD_SUBDIR
            )

        reward_lookup = self._load_rewards(rewards_dir)
        print(
            f"LabelledRewardPlugin: loaded {len(reward_lookup)} frame rewards "
            f"from {rewards_dir}"
        )
        return LabelledRewardPluginInstance(dataset, reward_lookup)

    @staticmethod
    def _load_rewards(rewards_dir: Path) -> Dict[int, float]:
        if not rewards_dir.exists():
            warnings.warn(
                f"Labelled rewards directory not found: {rewards_dir}. "
                "All rewards will default to 0.0."
            )
            return {}

        parquet_files = sorted(rewards_dir.glob("episode_*.parquet"))
        if not parquet_files:
            warnings.warn(
                f"No episode_*.parquet files in {rewards_dir}. "
                "All rewards will default to 0.0."
            )
            return {}

        dfs = [pd.read_parquet(f) for f in parquet_files]
        all_data = pd.concat(dfs, ignore_index=True)

        if "frame_index" not in all_data.columns or "reward" not in all_data.columns:
            raise ValueError(
                "Labelled reward parquet files must contain "
                "'frame_index' and 'reward' columns."
            )

        return dict(zip(all_data["frame_index"].astype(int), all_data["reward"].astype(float)))
