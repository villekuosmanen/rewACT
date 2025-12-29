from __future__ import annotations

import os
import sys
import einops
import torch
from torch import Tensor, nn
from .base import VisionEncoder


class VJepa2VisionEncoder(VisionEncoder):
    """V-JEPA 2 ViT -> patch tokens -> projection -> tokens with learned (patch + camera) position embeddings."""

    def __init__(
        self,
        *,
        dim_model: int,
        num_cameras: int,
        variant: str = "vit_large",
        weights: str,
        vit_patch_size: int = 16,
        pos_base_hw: tuple[int, int] = (30, 40),
    ) -> None:
        super().__init__()

        if num_cameras <= 0:
            raise ValueError(f"num_cameras must be >= 1. Got {num_cameras}.")
        self.num_cameras = num_cameras
        self.variant = variant
        self.vit_patch_size = vit_patch_size

        # Lazy import from ../vjepa2
        from pathlib import Path
        vjepa_path = str(Path(__file__).parents[3] / "vjepa2")
        if vjepa_path not in sys.path:
            sys.path.append(vjepa_path)

        try:
            from src.models import vision_transformer as vit_encoder
        except ImportError as e:
            raise ImportError(
                f"V-JEPA 2 is not available at {vjepa_path}. "
                "Ensure you have cloned it to the correct location."
            ) from e

        # V-JEPA 2 variants
        variant_to_ctor = {
            "vit_large": (vit_encoder.vit_large_rope, 1024),
            "vit_huge": (vit_encoder.vit_huge_rope, 1280),
            "vit_giant": (vit_encoder.vit_giant_xformers_rope, 1408),
        }
        if variant not in variant_to_ctor:
            raise ValueError(f"Unsupported V-JEPA 2 variant: {variant}. Supported: {sorted(variant_to_ctor.keys())}")

        ctor, embed_dim = variant_to_ctor[variant]

        # Initialize model for spatial mode (num_frames=1)
        self.vjepa_model = ctor(patch_size=vit_patch_size, num_frames=1)

        if not isinstance(weights, str) or len(weights) == 0:
            raise ValueError("vjepa2_weights must be a non-empty local checkpoint path.")
        if not os.path.exists(os.path.expanduser(weights)):
            raise FileNotFoundError(f"vjepa2_weights not found at: {weights}")

        # Load weights
        state_dict = torch.load(os.path.expanduser(weights), map_location="cpu", weights_only=True)["encoder"]
        # Clean state dict keys (same as in vjepa2/src/hub/backbones.py)
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.", "").replace("backbone.", "")
            clean_state_dict[new_k] = v
        self.vjepa_model.load_state_dict(clean_state_dict, strict=False)

        # Project V-JEPA embed_dim -> ACT dim_model
        self.proj = nn.Linear(embed_dim, dim_model)

        # Learned positional embeddings (DINOv3 style)
        h0, w0 = pos_base_hw
        self.patch_pos_base = nn.Parameter(torch.zeros(1, dim_model, h0, w0))
        nn.init.trunc_normal_(self.patch_pos_base, std=0.02)

        self.camera_embed = nn.Embedding(num_cameras, dim_model)
        nn.init.trunc_normal_(self.camera_embed.weight, std=0.02)

    def _make_pos_tokens(self, *, hp: int, wp: int, device, dtype, cam_idx: int) -> Tensor:
        pos = torch.nn.functional.interpolate(
            self.patch_pos_base.to(device=device, dtype=dtype),
            size=(hp, wp),
            mode="bicubic",
            align_corners=False,
        )
        pos = einops.rearrange(pos, "1 d h w -> (h w) 1 d")
        cam = self.camera_embed.weight[cam_idx].to(device=device, dtype=dtype).view(1, 1, -1)
        pos = pos + cam
        return pos.contiguous()

    def forward(self, img: Tensor, *, cam_idx: int = 0) -> tuple[Tensor, Tensor]:
        if not (0 <= cam_idx < self.num_cameras):
            raise ValueError(f"cam_idx out of range. Got {cam_idx} with num_cameras={self.num_cameras}.")

        # V-JEPA expects (B, 3, H, W) for 4D input or (B, 3, T, H, W) for 5D
        patch_tokens = self.vjepa_model(img)  # (B, N, embed_dim)

        tokens = self.proj(patch_tokens)  # (B, N, D)
        tokens = tokens.transpose(0, 1).contiguous()  # (N, B, D)

        _, _, h, w = img.shape
        hp = h // self.vit_patch_size
        wp = w // self.vit_patch_size

        pos_tokens = self._make_pos_tokens(
            hp=hp,
            wp=wp,
            device=tokens.device,
            dtype=tokens.dtype,
            cam_idx=cam_idx,
        )

        return tokens, pos_tokens

