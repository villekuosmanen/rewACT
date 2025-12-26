#!/usr/bin/env python

import argparse
import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rewact.vision_encoders import make_vision_encoder
from lerobot.configs.types import FeatureType, PolicyFeature


def smoke_test_resnet(device: str, batch_size: int, h: int, w: int, dim_model: int) -> None:
    cfg = SimpleNamespace(
        vision_encoder_type="resnet",
        freeze_vision_encoder=False,
        dim_model=dim_model,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        replace_final_stride_with_dilation=False,
    )
    enc = make_vision_encoder(cfg).to(device)
    img = torch.randn(batch_size, 3, h, w, device=device)
    tokens, pos = enc(img, cam_idx=0)

    assert pos.shape[0] == tokens.shape[0] and pos.shape[2] == tokens.shape[2], (tokens.shape, pos.shape)
    assert pos.shape[1] in (1, batch_size), pos.shape
    assert tokens.shape[1] == batch_size and tokens.shape[2] == dim_model, tokens.shape
    print(f"[resnet] tokens: {tuple(tokens.shape)}  pos: {tuple(pos.shape)}")


def smoke_test_dinov3(
    device: str,
    batch_size: int,
    h: int,
    w: int,
    dim_model: int,
    num_cameras: int,
    dinov3_variant: str,
    dinov3_weights: str | None,
) -> None:
    input_features = {
        f"observation.images.cam{i}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w))
        for i in range(num_cameras)
    }
    cfg = SimpleNamespace(
        vision_encoder_type="dinov3",
        freeze_vision_encoder=False,
        dim_model=dim_model,
        image_features=[f"cam{i}" for i in range(num_cameras)],
        input_features=input_features,
        dinov3_variant=dinov3_variant,
        dinov3_weights=dinov3_weights,
        dinov3_patch_size=16,
    )

    if not cfg.dinov3_weights:
        print("[dinov3] skipped (pass --dinov3-weights /path/to/ckpt.pth to run)")
        return

    try:
        enc = make_vision_encoder(cfg).to(device)
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"[dinov3] skipped: {e}")
        return

    for cam_idx in range(num_cameras):
        img = torch.randn(batch_size, 3, h, w, device=device)
        tokens, pos = enc(img, cam_idx=cam_idx)
        assert pos.shape[0] == tokens.shape[0] and pos.shape[2] == tokens.shape[2], (tokens.shape, pos.shape)
        assert pos.shape[1] in (1, batch_size), pos.shape
        assert tokens.shape[1] == batch_size and tokens.shape[2] == dim_model, tokens.shape
        print(f"[dinov3 cam={cam_idx}] tokens: {tuple(tokens.shape)}  pos: {tuple(pos.shape)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Quick shape smoke test for RewACT vision encoders.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--dim-model", type=int, default=512)
    p.add_argument("--h", type=int, default=224)
    p.add_argument("--w", type=int, default=224)
    p.add_argument("--num-cameras", type=int, default=2)
    p.add_argument(
        "--dinov3-variant",
        type=str,
        default="vitb16",
        choices=["vitb16", "vitl16", "convnext_base", "convnext_large"],
        help="Which DINOv3 backbone variant to instantiate (must match the checkpoint).",
    )
    p.add_argument(
        "--dinov3-weights",
        type=str,
        default=None,
        help="Local path to a DINOv3 checkpoint (must match --dinov3-variant). If provided, also runs the dinov3 check.",
    )
    args = p.parse_args()

    smoke_test_resnet(args.device, args.batch_size, args.h, args.w, args.dim_model)
    if args.dinov3_weights:
        smoke_test_dinov3(
            args.device,
            args.batch_size,
            args.h,
            args.w,
            args.dim_model,
            args.num_cameras,
            args.dinov3_variant,
            args.dinov3_weights,
        )


if __name__ == "__main__":
    main()


