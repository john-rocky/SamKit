#!/usr/bin/env python3
"""
Convert FastSAM (CASIA-IVA-Lab/FastSAM) to Core ML for SAMKit.

FastSAM is a YOLOv8-seg instance segmenter (not a SAM encoder/prompt-decoder), so it
segments everything in one forward pass; point / box prompts select among the instances
afterwards in `FastSamSession` (Swift). The exported model has one image input and four
split outputs:

  image       [1, 3, 640, 640]   float32, RGB, scaled to [0, 1] (letterboxed)
  boxes       [1, 4, 8400]        cx, cy, w, h in input (640) pixels
  scores      [1, 1, 8400]        single "object" class, sigmoid-calibrated
  mask_coeffs [1, 32, 8400]       per-anchor mask coefficients
  mask_protos [1, 32, 160, 160]   prototypes; instance mask = sigmoid(coeffs . protos)

Usage:
    pip install ultralytics coremltools torch
    python convert_fastsam_to_coreml.py            # both s and x
    python convert_fastsam_to_coreml.py --size s
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from coremltools.converters.mil.frontend.torch import ops as _ct_ops
from coremltools.converters.mil import Builder as mb


def _patched_int(context, node):
    """YOLOv8 Detect head casts dynamic shapes to int; force a const / plain cast."""
    inputs = _ct_ops._get_inputs(context, node)
    x = inputs[0]
    if x.val is not None:
        val = x.val
        if isinstance(val, np.ndarray):
            val = int(val.item()) if val.ndim == 0 else int(val.flat[0])
        else:
            val = int(val)
        res = mb.const(val=np.int32(val), name=node.name)
    else:
        res = mb.cast(x=x, dtype="int32", name=node.name)
    context.add(res)


_ct_ops._TORCH_OPS_REGISTRY.register_func(_patched_int, torch_alias=["int"], override=True)


FASTSAM_MODELS = {"s": "FastSAM-s.pt", "x": "FastSAM-x.pt"}
NUM_MASK_COEFFS = 32


class FastSAMWrapper(nn.Module):
    """Expose split boxes / scores / coeffs / protos from a YOLOv8-seg model."""

    def __init__(self, seg_model: nn.Module, num_classes: int):
        super().__init__()
        self.model = seg_model
        self.nc = num_classes

    def forward(self, image):
        out = self.model(image)
        # ultralytics 8.4: out[0] == (pred [1,4+nc+32,8400], protos [1,32,160,160]);
        # older builds returned pred at out[0] and protos as the last item of out[1].
        first = out[0]
        if isinstance(first, (tuple, list)):
            pred, protos = first[0], first[1]
        else:
            pred = first
            aux = out[1]
            protos = aux["proto"] if isinstance(aux, dict) else aux[-1]
        boxes = pred[:, :4, :]
        scores = pred[:, 4:4 + self.nc, :]
        mask_coeffs = pred[:, 4 + self.nc:, :]
        return boxes, scores, mask_coeffs, protos


def _image_input(input_size: int):
    try:
        layout = ct.colorlayout.RGB
    except AttributeError:
        layout = "RGB"
    return ct.ImageType(
        name="image",
        shape=(1, 3, input_size, input_size),
        scale=1.0 / 255.0,
        color_layout=layout,
    )


def convert(size_key: str, output_dir: Path, input_size: int = 640, tensor_input: bool = False):
    from ultralytics import FastSAM

    checkpoint = FASTSAM_MODELS[size_key]
    kind = "TensorType" if tensor_input else "ImageType (CVPixelBuffer)"
    print(f"=== Converting FastSAM-{size_key} @ {input_size}  [{kind}] ===")

    fs = FastSAM(checkpoint)
    seg_model = fs.model
    seg_model.eval()
    nc = int(getattr(seg_model.model[-1], "nc", 1))

    wrapper = FastSAMWrapper(seg_model, nc)
    wrapper.eval()

    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        _ = wrapper(dummy)
        _ = wrapper(dummy)

    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy, check_trace=False)

    print("  Converting to CoreML (FP16)...")
    inputs = ([ct.TensorType(name="image", shape=(1, 3, input_size, input_size))]
              if tensor_input else [_image_input(input_size)])
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[
            ct.TensorType(name="boxes"),
            ct.TensorType(name="scores"),
            ct.TensorType(name="mask_coeffs"),
            ct.TensorType(name="mask_protos"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.author = "SAMKit"
    mlmodel.short_description = f"FastSAM-{size_key} @ {input_size} (YOLOv8-seg) segment-everything"
    mlmodel.version = "1.0.0"

    out_path = output_dir / f"FastSAM_{size_key}_{input_size}.mlpackage"
    mlmodel.save(str(out_path))
    print(f"  Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert FastSAM to Core ML for SAMKit")
    parser.add_argument("--size", choices=["s", "x", "both"], default="s")
    parser.add_argument("--input-size", type=int, nargs="+", default=[640],
                        help="Pass multiple to produce a set, e.g. --input-size 320 512 640")
    parser.add_argument("--output", default=".")
    parser.add_argument("--tensor-input", action="store_true",
                        help="Use TensorType input instead of default ImageType")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for size_key in (["s", "x"] if args.size == "both" else [args.size]):
        for input_size in args.input_size:
            convert(size_key, output_dir, input_size, args.tensor_input)

    print("\nAdd the FastSAM_<s|x>_<size>.mlpackage(s) to your Xcode project or a Release.")


if __name__ == "__main__":
    main()
