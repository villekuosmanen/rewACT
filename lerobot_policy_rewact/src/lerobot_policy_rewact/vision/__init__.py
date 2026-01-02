from __future__ import annotations

from .base import VisionEncoder, _infer_pos_base_hw


def make_vision_encoder(config) -> VisionEncoder:
    """Factory for RewACT/ACT configs.

    Kept permissive w.r.t. config type so we can call it from RewACTConfig (subclass of ACTConfig).
    """

    vision_type = getattr(config, "vision_encoder_type", "resnet")
    freeze = bool(getattr(config, "freeze_vision_encoder", False))

    if vision_type == "resnet":
        from .resnet import ResNetVisionEncoder
        vision_encoder: VisionEncoder = ResNetVisionEncoder(
            dim_model=int(config.dim_model),
            vision_backbone=str(config.vision_backbone),
            pretrained_backbone_weights=getattr(config, "pretrained_backbone_weights", None),
            replace_final_stride_with_dilation=bool(getattr(config, "replace_final_stride_with_dilation", False)),
        )
    elif vision_type == "dinov3":
        from .dinov3 import DinoV3VisionEncoder
        num_cameras = len(getattr(config, "image_features", []))
        c = config.dinov3
        if c is None:
            raise ValueError("vision_encoder_type='dinov3' requires config.dinov3 to be set.")
        if c.weights is None:
            raise ValueError("vision_encoder_type='dinov3' requires config.dinov3.weights (local checkpoint path).")
        vision_encoder = DinoV3VisionEncoder(
            dim_model=int(config.dim_model),
            num_cameras=num_cameras,
            variant=c.variant,
            weights=str(c.weights),
            vit_patch_size=c.patch_size,
            pos_base_hw=_infer_pos_base_hw(config, vit_patch_size=c.patch_size),
            use_learned_pos_embed=getattr(c, "use_learned_pos_embed", False),
            use_patch_merge=getattr(c, "use_patch_merge", False),
        )
    elif vision_type == "vjepa2":
        from .vjepa2 import VJepa2VisionEncoder
        num_cameras = len(getattr(config, "image_features", []))
        c = config.vjepa2
        if c is None:
            raise ValueError("vision_encoder_type='vjepa2' requires config.vjepa2 to be set.")
        if c.weights is None:
            raise ValueError("vision_encoder_type='vjepa2' requires config.vjepa2.weights (local checkpoint path).")
        vision_encoder = VJepa2VisionEncoder(
            dim_model=int(config.dim_model),
            num_cameras=num_cameras,
            variant=c.variant,
            weights=str(c.weights),
            vit_patch_size=c.patch_size,
            pos_base_hw=_infer_pos_base_hw(config, vit_patch_size=c.patch_size),
            use_learned_pos_embed=getattr(c, "use_learned_pos_embed", False),
            use_patch_merge=getattr(c, "use_patch_merge", False),
        )
    else:
        raise ValueError(f"Unknown vision_encoder_type: {vision_type}")

    if freeze:
        vision_encoder.freeze_backbone(True)

    return vision_encoder
