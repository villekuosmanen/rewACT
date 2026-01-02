from __future__ import annotations

import einops
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.modeling_act import ACTSinusoidalPositionEmbedding2d
from .base import VisionEncoder


class ResNetVisionEncoder(VisionEncoder):
    """ResNet(layer4 feature map) -> flattened tokens with 2D sinusoidal positional embedding."""

    def __init__(
        self,
        *,
        dim_model: int,
        vision_backbone: str,
        pretrained_backbone_weights: str | None,
        replace_final_stride_with_dilation: bool,
    ) -> None:
        super().__init__()

        import torchvision  # local import so this module stays lightweight to import

        backbone_model = getattr(torchvision.models, vision_backbone)(
            replace_stride_with_dilation=[False, False, replace_final_stride_with_dilation],
            weights=pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        # Assumption: torchvision ResNet variants expose `layer4` as final feature map.
        self.resnet_feature_extractor = IntermediateLayerGetter(
            backbone_model, return_layers={"layer4": "feature_map"}
        )

        self.feat_proj = nn.Conv2d(backbone_model.fc.in_features, dim_model, kernel_size=1)
        self.pos_embed_2d = ACTSinusoidalPositionEmbedding2d(dim_model // 2)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze only the ResNet backbone. Keep projection layers trainable."""
        self.resnet_feature_extractor.requires_grad_(not freeze)
        if freeze:
            self.resnet_feature_extractor.eval()

    def forward(self, img: Tensor, *, cam_idx: int = 0) -> tuple[Tensor, Tensor]:
        feat = self.resnet_feature_extractor(img)["feature_map"]  # (B, Cb, H', W')
        pos = self.pos_embed_2d(feat).to(dtype=feat.dtype)  # (1, D, H', W')
        feat = self.feat_proj(feat)  # (B, D, H', W')

        tokens = einops.rearrange(feat, "b c h w -> (h w) b c")
        pos_tokens = einops.rearrange(pos, "b c h w -> (h w) b c")
        return tokens, pos_tokens
