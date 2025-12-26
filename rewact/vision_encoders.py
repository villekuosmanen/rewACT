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
        patch_size: int = 16,
        # Learned patch-pos is stored as a small base grid and interpolated to runtime Hp×Wp.
        pos_base_hw: tuple[int, int] = (30, 40),
    ) -> None:
        super().__init__()

        if num_cameras <= 0:
            raise ValueError(f"num_cameras must be >= 1. Got {num_cameras}.")
        self.num_cameras = num_cameras

        # Lazy import so ResNet users don't need dinov3 installed.
        try:
            from dinov3.hub.backbones import Weights as DinoHubWeights
            from dinov3.hub.backbones import dinov3_vitl16
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "DINOv3 is not available. Install the local repo, e.g. `pip install -e /home/user/Desktop/code/dinov3`."
            ) from e

        if variant != "vitl16":
            raise ValueError(f"Only variant='vitl16' is supported in v1. Got {variant}.")

        if not isinstance(weights, str) or len(weights) == 0:
            raise ValueError("dinov3_weights must be a non-empty local checkpoint path.")
        if not os.path.exists(os.path.expanduser(weights)):
            raise FileNotFoundError(f"dinov3_weights not found at: {weights}")

        # Note: DINOv3 hub loader returns a DinoVisionTransformer. `forward_features(x)` returns a dict
        # with `x_norm_patchtokens: (B, N_patches, embed_dim)`.
        self.dinov3_model = dinov3_vitl16(pretrained=True, weights=weights)

        # Sanity: keep track of patch size (hub model also has `patch_size` attribute).
        self.patch_size = patch_size

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

    def forward(self, img: Tensor, *, cam_idx: int = 0) -> tuple[Tensor, Tensor]:
        if not (0 <= cam_idx < self.num_cameras):
            raise ValueError(f"cam_idx out of range. Got {cam_idx} with num_cameras={self.num_cameras}.")

        _, _, h, w = img.shape
        # DINO patch embed expects dimensions divisible by patch size in most setups.
        hp = h // self.patch_size
        wp = w // self.patch_size

        out = self.dinov3_model.forward_features(img)
        if "x_norm_patchtokens" not in out:  # pragma: no cover
            raise RuntimeError(f"Unexpected DINOv3 forward_features output keys: {list(out.keys())}")
        patch_tokens = out["x_norm_patchtokens"]  # (B, N, embed_dim)

        tokens = self.proj(patch_tokens)  # (B, N, D)
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
                f"(Hp={hp}, Wp={wp}, patch_size={self.patch_size})."
            )

        return tokens, pos_tokens


def _infer_dinov3_pos_base_hw(config, *, patch_size: int) -> tuple[int, int]:
    """Infer a sensible base grid (Hp, Wp) from configured image shapes. """
    img_feats = getattr(config, "image_features", None)
    if isinstance(img_feats, dict) and len(img_feats) > 0:
        ft = next(iter(img_feats.values()))
        shape = getattr(ft, "shape", None)
        if isinstance(shape, tuple) and len(shape) >= 3:
            # Common: (3, H, W)
            h = int(shape[-2])
            w = int(shape[-1])
            return max(1, h // patch_size), max(1, w // patch_size)
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
        patch_size = int(getattr(config, "dinov3_patch_size", 16))
        weights = getattr(config, "dinov3_weights", None)
        if weights is None:
            raise ValueError("vision_encoder_type='dinov3' requires config.dinov3_weights (local checkpoint path).")
        enc = DinoV3VisionEncoder(
            dim_model=int(config.dim_model),
            num_cameras=num_cameras,
            variant=str(getattr(config, "dinov3_variant", "vitl16")),
            weights=str(weights),
            patch_size=patch_size,
            pos_base_hw=_infer_dinov3_pos_base_hw(config, patch_size=patch_size),
        )
    else:
        raise ValueError(f"Unknown vision_encoder_type: {vision_type}")

    if freeze:
        enc.requires_grad_(False)
        enc.eval()

    return enc


