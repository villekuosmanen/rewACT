from __future__ import annotations

import os
import einops
import torch
from torch import Tensor, nn
from lerobot.policies.act.modeling_act import ACTSinusoidalPositionEmbedding2d
from .base import VisionEncoder


class DinoV3VisionEncoder(VisionEncoder):
    """DINOv3 ViT -> patch tokens -> projection -> tokens with learned (patch + camera) position embeddings."""

    def __init__(
        self,
        *,
        dim_model: int,
        num_cameras: int,
        variant: str = "vitl16",
        weights: str,
        vit_patch_size: int = 16,
        # Learned patch-pos is stored as a small base grid and interpolated to runtime Hp×Wp.
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

        # Lazy import so ResNet users don't need dinov3 installed.
        try:
            from dinov3.hub.backbones import (
                dinov3_convnext_base,
                dinov3_convnext_large,
                dinov3_vitb16,
                dinov3_vitl16,
            )
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "DINOv3 is not available. Install the local repo, e.g. `pip install -e /home/user/Desktop/code/dinov3`."
            ) from e

        variant_to_constructor = {
            "vitb16": (dinov3_vitb16, "vit"),
            "vitl16": (dinov3_vitl16, "vit"),
            "convnext_base": (dinov3_convnext_base, "convnext"),
            "convnext_large": (dinov3_convnext_large, "convnext"),
        }
        if variant not in variant_to_constructor:
            raise ValueError(f"Unsupported DINOv3 variant: {variant}. Supported: {sorted(variant_to_constructor.keys())}")
        constructor, grid_kind = variant_to_constructor[variant]
        self.variant = variant
        self.grid_kind = grid_kind

        if not isinstance(weights, str) or len(weights) == 0:
            raise ValueError("dinov3_weights must be a non-empty local checkpoint path.")
        if not os.path.exists(os.path.expanduser(weights)):
            raise FileNotFoundError(f"dinov3_weights not found at: {weights}")

        # Note: DINOv3 hub loader returns a DinoVisionTransformer. `forward_features(x)` returns a dict
        # with `x_norm_patchtokens: (B, N_patches, embed_dim)`.
        self.dinov3_model = constructor(pretrained=True, weights=weights)

        # For ViT variants only.
        self.vit_patch_size = int(vit_patch_size)

        # Project DINO embed_dim -> ACT dim_model.
        embed_dim = getattr(self.dinov3_model, "embed_dim", None)
        if embed_dim is None:  # pragma: no cover
            raise RuntimeError("Unexpected DINOv3 model: missing `embed_dim`.")
        self.proj = nn.Linear(int(embed_dim), dim_model)

        if use_patch_merge:
            # Patch merging: concat 2x2 (4 tokens) then project back to dim_model
            self.merger_linear = nn.Linear(dim_model * 4, dim_model)

        if use_learned_pos_embed:
            # Learned positional embeddings.
            h0, w0 = pos_base_hw
            if h0 <= 0 or w0 <= 0:
                raise ValueError(f"pos_base_hw must be positive. Got {pos_base_hw}.")
            self.patch_pos_base = nn.Parameter(torch.zeros(1, dim_model, h0, w0))
            nn.init.trunc_normal_(self.patch_pos_base, std=0.02)

            self.camera_embed = nn.Embedding(num_cameras, dim_model)
            nn.init.trunc_normal_(self.camera_embed.weight, std=0.02)
        else:
            self.pos_embed_2d = ACTSinusoidalPositionEmbedding2d(dim_model // 2)

    def _make_pos_tokens(self, *, hp: int, wp: int, device, dtype, cam_idx: int) -> Tensor:
        if self.use_learned_pos_embed:
            # Interpolate base grid -> (1, D, Hp, Wp), then flatten to (Hp*Wp, 1, D).
            pos = torch.nn.functional.interpolate(
                self.patch_pos_base.to(device=device, dtype=dtype),
                size=(hp, wp),
                mode="bicubic",
                align_corners=False,
            )
            pos = einops.rearrange(pos, "1 d h w -> (h w) 1 d")
            cam = self.camera_embed.weight[cam_idx].to(device=device, dtype=dtype).view(1, 1, -1)
            pos = pos + cam  # broadcast cam over patches
            return pos.contiguous()
        else:
            # Simple sinusoidal embeddings (ignoring cam_idx)
            # Create a dummy tensor of the correct spatial shape to get pos embeddings
            dummy = torch.zeros((1, self.dim_model, hp, wp), device=device, dtype=dtype)
            pos = self.pos_embed_2d(dummy)  # (1, D, Hp, Wp)
            pos = einops.rearrange(pos, "1 d h w -> (h w) 1 d")
            return pos.contiguous()

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze only the DINOv3 backbone. Keep projection and pos-embed layers trainable."""
        self.dinov3_model.requires_grad_(not freeze)
        if freeze:
            self.dinov3_model.eval()

    @staticmethod
    def _infer_hw_from_seq_len(n_tokens: int, *, h: int, w: int) -> tuple[int, int]:
        """Infer (Hp, Wp) such that Hp*Wp == n_tokens and Hp/Wp ~= h/w."""
        if n_tokens <= 0:
            raise ValueError(f"n_tokens must be positive. Got {n_tokens}.")
        # Find factor pair closest to aspect ratio.
        target_ratio = float(h) / float(w)
        best = (1, n_tokens)
        best_err = float("inf")
        # Iterate over divisors up to sqrt(n_tokens).
        r = int(n_tokens**0.5)
        for hp in range(1, r + 1):
            if n_tokens % hp != 0:
                continue
            wp = n_tokens // hp
            err = abs((hp / wp) - target_ratio)
            if err < best_err:
                best_err = err
                best = (hp, wp)
        return best

    def forward(self, img: Tensor, *, cam_idx: int = 0) -> tuple[Tensor, Tensor]:
        if not (0 <= cam_idx < self.num_cameras):
            raise ValueError(f"cam_idx out of range. Got {cam_idx} with num_cameras={self.num_cameras}.")

        out = self.dinov3_model.forward_features(img)
        if "x_norm_patchtokens" not in out:  # pragma: no cover
            raise RuntimeError(f"Unexpected DINOv3 forward_features output keys: {list(out.keys())}")
        patch_tokens = out["x_norm_patchtokens"]  # (B, N, embed_dim)

        tokens = self.proj(patch_tokens)  # (B, N, D)

        # Determine a 2D token grid (Hp, Wp) for learned positional embeddings.
        _, _, h, w = img.shape
        if self.grid_kind == "vit":
            hp = h // self.vit_patch_size
            wp = w // self.vit_patch_size
        else:
            hp, wp = self._infer_hw_from_seq_len(tokens.shape[1], h=h, w=w)

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

        # Safety: token count must match computed Hp*Wp.
        if tokens.shape[0] != pos_tokens.shape[0]:  # pragma: no cover
            raise RuntimeError(
                f"DINO tokens length mismatch. Got N={tokens.shape[0]} but Hp*Wp={pos_tokens.shape[0]} "
                f"(Hp={hp}, Wp={wp}, variant={self.variant})."
            )

        return tokens, pos_tokens
