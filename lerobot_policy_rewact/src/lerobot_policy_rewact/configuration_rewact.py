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


@PreTrainedConfig.register_subclass("rewact")
@dataclass
class RewACTConfig(ACTConfig):
    # Reward prediction head
    use_reward_head: bool = True # Deprecated - unused
    reward_loss_weight: float = 0.1
    num_value_bins: int = 100  # Number of bins for distributional value function
    value_min: int = 0
    value_max: int = 1

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

    def __post_init__(self):
        # Keep `pretrained_backbone_weights` aligned with the selected vision encoder.
        # - For resnet, this remains the torchvision "weights" enum string (inherited from ACTConfig).
        # - For other encoders, this is not consumed by the model builders, but it *is* printed in
        #   config dumps; we set it to the actual checkpoint path to avoid confusion.
        if self.vision_encoder_type == "dinov3" and self.dinov3:
            self.pretrained_backbone_weights = self.dinov3.weights
        elif self.vision_encoder_type == "vjepa2" and self.vjepa2:
            self.pretrained_backbone_weights = self.vjepa2.weights
        elif self.vision_encoder_type == "sam3" and self.sam3:
            self.pretrained_backbone_weights = self.sam3.weights

        # Set vision_backbone to reflect the actual encoder being used
        if self.vision_encoder_type == "dinov3" and self.dinov3:
            self.vision_backbone = f"dinov3_{self.dinov3.variant}"
        elif self.vision_encoder_type == "vjepa2" and self.vjepa2:
            self.vision_backbone = f"vjepa2_{self.vjepa2.variant}"
        elif self.vision_encoder_type == "sam3" and self.sam3:
            self.vision_backbone = f"sam3_{self.sam3.variant}"

    @property
    def temporal_offset(self) -> int:
        """Return temporal_offset for VJEPA2, else 0."""
        if self.vision_encoder_type == "vjepa2" and self.vjepa2:
            return self.vjepa2.temporal_offset
        return 0
