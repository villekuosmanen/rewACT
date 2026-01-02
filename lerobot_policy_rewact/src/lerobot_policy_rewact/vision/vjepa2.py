from __future__ import annotations

import os
import sys
import einops
import torch
from torch import Tensor, nn
from lerobot.policies.act.modeling_act import ACTSinusoidalPositionEmbedding2d
from .base import VisionEncoder


class VJepa2VisionEncoder(VisionEncoder):
    """V-JEPA 2 video ViT: (B,3,T,H,W) -> patch tokens -> projection -> learned (patch + camera) pos embeddings."""

    def __init__(
        self,
        *,
        dim_model: int,
        num_cameras: int,
        variant: str = "vit_large",
        weights: str,                  # path to the V-JEPA 2 checkpoint
        vit_patch_size: int = 16,
        tubelet_size: int = 2,      # number of frames in the video tubelet
        pos_base_hw: tuple[int, int] = (30, 40),
        use_learned_pos_embed: bool = False,
        use_patch_merge: bool = False,
    ) -> None:
        super().__init__()

        if num_cameras <= 0:
            raise ValueError(f"num_cameras must be >= 1. Got {num_cameras}.")
        self.num_cameras = num_cameras
        self.dim_model = dim_model
        self.use_learned_pos_embed = use_learned_pos_embed
        self.use_patch_merge = use_patch_merge
        self.variant = variant
        self.vit_patch_size = vit_patch_size
        self.tubelet_size = tubelet_size

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
        variant_to_constructor = {
            "vit_large": (vit_encoder.vit_large_rope, 1024),
            "vit_huge": (vit_encoder.vit_huge_rope, 1280),
            "vit_giant": (vit_encoder.vit_giant_xformers_rope, 1408),
        }
        if variant not in variant_to_constructor:
            raise ValueError(f"Unsupported V-JEPA 2 variant: {variant}. Supported: {sorted(variant_to_constructor.keys())}")

        constructor, embed_dim = variant_to_constructor[variant]

        # Initialize model matching checkpoint (video mode with tubelet_size frames)
        self.vjepa_model = constructor(patch_size=vit_patch_size, num_frames=tubelet_size, tubelet_size=tubelet_size)

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

        if use_patch_merge:
            # Patch merging: concat 2x2 (4 tokens) then project back to dim_model
            self.merger_linear = nn.Linear(dim_model * 4, dim_model)

        if use_learned_pos_embed:
            # Learned positional embeddings (DINOv3 style)
            h0, w0 = pos_base_hw
            self.patch_pos_base = nn.Parameter(torch.zeros(1, dim_model, h0, w0))
            nn.init.trunc_normal_(self.patch_pos_base, std=0.02)

            self.camera_embed = nn.Embedding(num_cameras, dim_model)
            nn.init.trunc_normal_(self.camera_embed.weight, std=0.02)
        else:
            self.pos_embed_2d = ACTSinusoidalPositionEmbedding2d(dim_model // 2)

    def _make_pos_tokens(self, *, hp: int, wp: int, device, dtype, cam_idx: int) -> Tensor:
        if self.use_learned_pos_embed:
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
        else:
            # Simple sinusoidal embeddings (ignoring cam_idx)
            dummy = torch.zeros((1, self.dim_model, hp, wp), device=device, dtype=dtype)
            pos = self.pos_embed_2d(dummy)
            pos = einops.rearrange(pos, "1 d h w -> (h w) 1 d")
            return pos.contiguous()

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze only the V-JEPA 2 backbone. Keep projection and pos-embed layers trainable."""
        self.vjepa_model.requires_grad_(not freeze)
        if freeze:
            self.vjepa_model.eval()

    def forward(self, img: Tensor, *, cam_idx: int = 0) -> tuple[Tensor, Tensor]:
        """
        Args:
            img: (B, 3, T, H, W) video tensor where T >= tubelet_size
            cam_idx: camera index for positional embedding
        """
        if not (0 <= cam_idx < self.num_cameras):
            raise ValueError(f"cam_idx out of range. Got {cam_idx} with num_cameras={self.num_cameras}.")

        _, _, _, h, w = img.shape
        patch_tokens = self.vjepa_model(img)  # (B, N, embed_dim)

        tokens = self.proj(patch_tokens)  # (B, N, D)
        hp = h // self.vit_patch_size
        wp = w // self.vit_patch_size

        if self.use_patch_merge:
            # 1. Reshape to grid: (B, N, D) -> (B, D, Hp, Wp)
            tokens = einops.rearrange(tokens, "b (h w) d -> b d h w", h=hp, w=wp)
            # 2. Patch Merge (Space-to-Depth): (B, D, Hp, Wp) -> (B, 4D, Hp/2, Wp/2)
            tokens = einops.rearrange(tokens, "b d (h p1) (w p2) -> b (d p1 p2) h w", p1=2, p2=2)
            # 3. Project back to dim_model: (B, 4D, H', W') -> (B, H', W', 4D) -> (B, H', W', D)
            tokens = self.merger_linear(tokens.permute(0, 2, 3, 1))
            # 4. Flatten back: (B, H', W', D) -> (H'*W', B, D)
            hp, wp = hp // 2, wp // 2
            tokens = einops.rearrange(tokens, "b h w d -> (h w) b d")
        else:
            tokens = tokens.transpose(0, 1).contiguous()  # (N, B, D)

        pos_tokens = self._make_pos_tokens(
            hp=hp,
            wp=wp,
            device=tokens.device,
            dtype=tokens.dtype,
            cam_idx=cam_idx,
        )

        return tokens, pos_tokens
