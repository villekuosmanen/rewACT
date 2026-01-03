# Copyright 2024 Tony Z. Zhao, The HuggingFace Inc. team, and Ville Kuosmanen. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("actvantage")
@dataclass
class ACTvantageConfig(ACTConfig):
    advantage_dropout_prob: float = 0.3
    value_min: int = 0
    value_max: int = 1
@dataclass
class DinoV3Config:
    variant: str = "vitl16"
    # Path to a local DINOv3 checkpoint.
    weights: str | None = None
    patch_size: int = 16
    # Placeholder for future (not wired yet):
    preprocess: str = "none"
    use_learned_pos_embed: bool = False
    use_patch_merge: bool = False


@dataclass
class VJepa2Config:
    variant: str = "vit_large"
    # Path to a local V-JEPA 2 checkpoint.
    weights: str | None = None
    patch_size: int = 16
    # Frames to look back for temporal context (e.g., 30 = 1 sec at 30fps)
    temporal_offset: int = 30
    use_learned_pos_embed: bool = False
    use_patch_merge: bool = False


@dataclass
class SAM3Config:
    variant: str = "vit_l"
    # Path to a local SAM 3 checkpoint.
    weights: str | None = None
