from __future__ import annotations

import torch
from torch import Tensor, nn


class VisionEncoder(nn.Module):
    """Image -> (tokens, pos_tokens) adapter.

    Contract:
      - img: (B, 3, H, W)
      - tokens: (S_img, B, dim_model)
      - pos_tokens: (S_img, 1, dim_model) (broadcastable across batch)

    Note: ACT's encoder layer literally does `q=k=x+pos_embed` when pos_embed is provided, so
    for simplicity and to avoid special-casing downstream, adapters should generally always
    return a pos_tokens tensor (not None).
    """

    def forward(self, img: Tensor, *, cam_idx: int = 0) -> tuple[Tensor, Tensor]:  # pragma: no cover
        raise NotImplementedError

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze the heavy vision backbone. 
        Projection layers and positional embeddings should generally remain trainable.
        """
        # Default implementation: freeze everything (can be overridden)
        self.requires_grad_(not freeze)
        if freeze:
            self.eval()


def _infer_pos_base_hw(config, *, vit_patch_size: int) -> tuple[int, int]:
    """Infer a sensible base grid (Hp, Wp) from configured image shapes. """
    img_feats = getattr(config, "image_features", None)
    if isinstance(img_feats, (dict, list)) and len(img_feats) > 0:
        if isinstance(img_feats, dict):
            ft = next(iter(img_feats.values()))
        else:
            # If it's a list, we might need to look into input_features
            input_feats = getattr(config, "input_features", {})
            feat_name = f"observation.images.{img_feats[0]}"
            ft = input_feats.get(feat_name)
            
        if ft is not None:
            shape = getattr(ft, "shape", None)
            if isinstance(shape, tuple) and len(shape) >= 3:
                # Common: (3, H, W)
                h = int(shape[-2])
                w = int(shape[-1])
                # Special case for ConvNeXt which has an effective stride of 32
                if hasattr(config, "dinov3") and config.dinov3 is not None and config.dinov3.variant.startswith("convnext"):
                    return max(1, h // 32), max(1, w // 32)
                return max(1, h // vit_patch_size), max(1, w // vit_patch_size)
                
    # Fallback that works well for 480×640 with patch=16 (30×40).
    return (30, 40)
