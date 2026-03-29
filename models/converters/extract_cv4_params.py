#!/usr/bin/env python3
"""
Extract BNContrastiveHead parameters from YOLO-World for iOS inference.

The yolov8s-worldv2 model uses BNContrastiveHead per detection level:
    x = BatchNorm2d(features)       # 512-channel normalization
    w = F.normalize(text, dim=-1)   # L2-normalize text embeddings
    logits = einsum(x, w) * exp(logit_scale) + bias

The CoreML model outputs raw cv3 features (pre-BN). This script exports
fused BN parameters so Swift can replicate the BNContrastiveHead math:
    bn_scale[c] = weight[c] / sqrt(running_var[c] + eps)
    bn_offset[c] = bias_bn[c] - running_mean[c] * bn_scale[c]
    BN(x)[c] = x[c] * bn_scale[c] + bn_offset[c]

Usage:
    python extract_cv4_params.py --output ../manifests/grounding/cv4_params.json
"""

import argparse
import json
import numpy as np
from pathlib import Path

# Model size variants
YOLO_WORLD_MODELS = {
    "s": "yolov8s-worldv2",
    "m": "yolov8m-worldv2",
    "l": "yolov8l-worldv2",
    "x": "yolov8x-worldv2",
}


def extract_params(model_name: str = "yolov8l-worldv2"):
    """Extract BNContrastiveHead parameters from YOLO-World model."""
    from ultralytics import YOLO

    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    detect = model.model.model[-1]
    print(f"Detect head: {type(detect).__name__}, levels: {detect.nl}")

    levels = []
    for i in range(detect.nl):
        head = detect.cv4[i]
        logit_scale = head.logit_scale.item()
        bias = head.bias.item()

        bn = head.norm
        w = bn.weight.detach().cpu().numpy()
        b = bn.bias.detach().cpu().numpy()
        m = bn.running_mean.detach().cpu().numpy()
        v = bn.running_var.detach().cpu().numpy()

        bn_scale = w / np.sqrt(v + bn.eps)
        bn_offset = b - m * bn_scale

        # Verify fused computation matches original
        dummy = np.random.randn(len(w)).astype(np.float32)
        original = (dummy - m) / np.sqrt(v + bn.eps) * w + b
        fused = dummy * bn_scale + bn_offset
        max_diff = np.max(np.abs(original - fused))

        print(f"Level {i}: logit_scale={logit_scale:.4f} (exp={np.exp(logit_scale):.4f}), bias={bias:.4f}")
        print(f"  bn_scale: [{bn_scale.min():.3f}, {bn_scale.max():.3f}] mean={bn_scale.mean():.3f}")
        print(f"  bn_offset: [{bn_offset.min():.3f}, {bn_offset.max():.3f}] mean={bn_offset.mean():.3f}")
        print(f"  fused verification: max_diff={max_diff:.8f}")

        levels.append({
            "logit_scale": float(logit_scale),
            "bias": float(bias),
            "bn_scale": [round(float(x), 6) for x in bn_scale],
            "bn_offset": [round(float(x), 6) for x in bn_offset],
        })

    strides = detect.stride.cpu().numpy().astype(int).tolist()
    input_size = 640
    anchors_per_level = [(input_size // s) ** 2 for s in strides]
    print(f"Anchors per level: {anchors_per_level} (total {sum(anchors_per_level)})")

    return {
        "anchors_per_level": anchors_per_level,
        "levels": levels,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract BNContrastiveHead parameters for iOS inference"
    )
    parser.add_argument(
        "--size", type=str, default=None, choices=["s", "m", "l", "x"],
        help="Model size shorthand: s(mall), m(edium), l(arge), x(tra-large)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="YOLO-World model name (overrides --size)",
    )
    parser.add_argument(
        "--output", type=str, default="../manifests/grounding/cv4_params.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    if args.model:
        model_name = args.model
    elif args.size:
        model_name = YOLO_WORLD_MODELS[args.size]
    else:
        model_name = YOLO_WORLD_MODELS["l"]

    params = extract_params(model_name)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(params, f)

    print(f"\nSaved to {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
