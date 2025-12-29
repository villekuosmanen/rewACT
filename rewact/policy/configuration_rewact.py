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


@dataclass
class DinoV3Config:
    variant: str = "vitl16"
    # Path to a local DINOv3 checkpoint.
    weights: str | None = None
    patch_size: int = 16
    # Placeholder for future (not wired yet):
    preprocess: str = "none"


@dataclass
class VJepa2Config:
    variant: str = "vit_large"
    # Path to a local V-JEPA 2 checkpoint.
    weights: str | None = None
    patch_size: int = 16


@dataclass
class SAM3Config:
    variant: str = "vit_l"
    # Path to a local SAM 3 checkpoint.
    weights: str | None = None


@PreTrainedConfig.register_subclass("rewact")
@dataclass
class RewACTConfig(ACTConfig):
    # Reward prediction head
    use_reward_head: bool = True
    reward_loss_weight: float = 2.0

    # Vision encoder abstraction.
    # - "resnet": torchvision resnet -> feature map -> flatten tokens (current behavior)
    # - "dinov3": dinov3 ViT-L/16 -> patch tokens
    # - "vjepa2": v-jepa 2 video vit -> spatial tokens
    # - "sam3": sam 3 perception encoder -> unconditioned tokens
    vision_encoder_type: str = "resnet"
    freeze_vision_encoder: bool = True

    # Nested backbone configurations (None when not in use)
    dinov3: Optional[DinoV3Config] = None
    vjepa2: Optional[VJepa2Config] = None
    sam3: Optional[SAM3Config] = None
