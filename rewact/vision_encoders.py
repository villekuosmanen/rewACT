from __future__ import annotations

import os

import einops
import torch
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.modeling_act import ACTSinusoidalPositionEmbedding2d


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

    def forward(self, img: Tensor, *, cam_idx: int = 0) -> tuple[Tensor, Tensor]:
        feat = self.resnet_feature_extractor(img)["feature_map"]  # (B, Cb, H', W')
        pos = self.pos_embed_2d(feat).to(dtype=feat.dtype)  # (1, D, H', W')
        feat = self.feat_proj(feat)  # (B, D, H', W')

        tokens = einops.rearrange(feat, "b c h w -> (h w) b c")
        pos_tokens = einops.rearrange(pos, "b c h w -> (h w) b c")
        return tokens, pos_tokens


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
    ) -> None:
        super().__init__()

        if num_cameras <= 0:
            raise ValueError(f"num_cameras must be >= 1. Got {num_cameras}.")
        self.num_cameras = num_cameras

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

        variant_to_ctor = {
            "vitb16": (dinov3_vitb16, "vit"),
            "vitl16": (dinov3_vitl16, "vit"),
            "convnext_base": (dinov3_convnext_base, "convnext"),
            "convnext_large": (dinov3_convnext_large, "convnext"),
        }
        if variant not in variant_to_ctor:
            raise ValueError(f"Unsupported DINOv3 variant: {variant}. Supported: {sorted(variant_to_ctor.keys())}")
        ctor, grid_kind = variant_to_ctor[variant]
        self.variant = variant
        self.grid_kind = grid_kind

        if not isinstance(weights, str) or len(weights) == 0:
            raise ValueError("dinov3_weights must be a non-empty local checkpoint path.")
        if not os.path.exists(os.path.expanduser(weights)):
            raise FileNotFoundError(f"dinov3_weights not found at: {weights}")

        # Note: DINOv3 hub loader returns a DinoVisionTransformer. `forward_features(x)` returns a dict
        # with `x_norm_patchtokens: (B, N_patches, embed_dim)`.
        self.dinov3_model = ctor(pretrained=True, weights=weights)

        # For ViT variants only.
        self.vit_patch_size = int(vit_patch_size)

        # Project DINO embed_dim -> ACT dim_model.
        embed_dim = getattr(self.dinov3_model, "embed_dim", None)
        if embed_dim is None:  # pragma: no cover
            raise RuntimeError("Unexpected DINOv3 model: missing `embed_dim`.")
        self.proj = nn.Linear(int(embed_dim), dim_model)

        # Learned positional embeddings.
        h0, w0 = pos_base_hw
        if h0 <= 0 or w0 <= 0:
            raise ValueError(f"pos_base_hw must be positive. Got {pos_base_hw}.")
        self.patch_pos_base = nn.Parameter(torch.zeros(1, dim_model, h0, w0))
        nn.init.trunc_normal_(self.patch_pos_base, std=0.02)

        self.camera_embed = nn.Embedding(num_cameras, dim_model)
        nn.init.trunc_normal_(self.camera_embed.weight, std=0.02)

    def _make_pos_tokens(self, *, hp: int, wp: int, device, dtype, cam_idx: int) -> Tensor:
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

    @staticmethod
    def _infer_hw_from_seq_len(n_tokens: int, *, h: int, w: int) -> tuple[int, int]:
        """Infer (Hp, Wp) such that Hp*Wp == n_tokens and Hp/Wp ~= h/w.

        This is used for non-ViT backbones (e.g. ConvNeXt) where token grids come from strided feature maps
        and exact output sizes can depend on conv padding/stride details.
        """
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
        tokens = tokens.transpose(0, 1).contiguous()  # (N, B, D)

        # Determine a 2D token grid (Hp, Wp) for learned positional embeddings.
        _, _, h, w = img.shape
        if self.grid_kind == "vit":
            hp = h // self.vit_patch_size
            wp = w // self.vit_patch_size
        else:
            hp, wp = self._infer_hw_from_seq_len(tokens.shape[0], h=h, w=w)

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


def _infer_dinov3_pos_base_hw(config, *, variant: str, vit_patch_size: int) -> tuple[int, int]:
    """Infer a sensible base grid (Hp, Wp) from configured image shapes. """
    img_feats = getattr(config, "image_features", None)
    if isinstance(img_feats, dict) and len(img_feats) > 0:
        ft = next(iter(img_feats.values()))
        shape = getattr(ft, "shape", None)
        if isinstance(shape, tuple) and len(shape) >= 3:
            # Common: (3, H, W)
            h = int(shape[-2])
            w = int(shape[-1])
            if variant.startswith("convnext"):
                # ConvNeXt has an effective stride of 32 (4 then 2×2×2).
                return max(1, h // 32), max(1, w // 32)
            return max(1, h // vit_patch_size), max(1, w // vit_patch_size)
    # Fallback that works well for 480×640 with patch=16 (30×40).
    return (30, 40)


def make_vision_encoder(config) -> VisionEncoder:
    """Factory for RewACT/ACT configs.

    Kept permissive w.r.t. config type so we can call it from RewACTConfig (subclass of ACTConfig).
    """

    vision_type = getattr(config, "vision_encoder_type", "resnet")
    freeze = bool(getattr(config, "freeze_vision_encoder", False))

    if vision_type == "resnet":
        enc: VisionEncoder = ResNetVisionEncoder(
            dim_model=int(config.dim_model),
            vision_backbone=str(config.vision_backbone),
            pretrained_backbone_weights=getattr(config, "pretrained_backbone_weights", None),
            replace_final_stride_with_dilation=bool(getattr(config, "replace_final_stride_with_dilation", False)),
        )
    elif vision_type == "dinov3":
        img_feats = getattr(config, "image_features", [])
        num_cameras = len(img_feats)
        vit_patch_size = int(getattr(config, "dinov3_patch_size", 16))
        weights = getattr(config, "dinov3_weights", None)
        if weights is None:
            raise ValueError("vision_encoder_type='dinov3' requires config.dinov3_weights (local checkpoint path).")
        variant = str(getattr(config, "dinov3_variant", "vitl16"))
        enc = DinoV3VisionEncoder(
            dim_model=int(config.dim_model),
            num_cameras=num_cameras,
            variant=variant,
            weights=str(weights),
            vit_patch_size=vit_patch_size,
            pos_base_hw=_infer_dinov3_pos_base_hw(config, variant=variant, vit_patch_size=vit_patch_size),
        )
    else:
        raise ValueError(f"Unknown vision_encoder_type: {vision_type}")

    if freeze:
        enc.requires_grad_(False)
        enc.eval()

    return enc


