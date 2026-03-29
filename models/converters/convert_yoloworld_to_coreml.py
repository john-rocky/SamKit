#!/usr/bin/env python3
"""
Convert YOLO-World model to Core ML format for open-vocabulary object detection.

Produces four outputs:
  1. yoloworld_detector.mlpackage - Visual detector (boxes + scores outputs)
  2. clip_text_encoder.mlpackage  - CLIP text encoder (any text → embeddings)
  3. clip_vocab.json              - BPE vocabulary for Swift tokenizer
  4. cv4_params.json              - BNContrastiveHead parameters (reference)

Architecture note:
  The CoreML detector includes the full BNContrastiveHead scoring pipeline
  internally, so output scores are already sigmoid-calibrated confidence values.
  The model outputs separate "boxes" [1,4,8400] and "scores" [1,NC,8400] tensors.

Usage:
    python convert_yoloworld_to_coreml.py --output ../manifests/grounding
    python convert_yoloworld_to_coreml.py --size m --output ../manifests/grounding
"""

import argparse
import json
import hashlib
import gzip
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coremltools as ct


# Maximum number of class slots (detector input dimension)
MAX_CLASSES = 80
CONTEXT_LENGTH = 77

# Model size variants: all use CLIP ViT-B/32 (512-dim), differ in visual backbone
YOLO_WORLD_MODELS = {
    "s": "yolov8s-worldv2",   # ~25 MB CoreML, COCO AP ~38.7
    "m": "yolov8m-worldv2",   # ~50 MB CoreML, COCO AP ~43.3
    "l": "yolov8l-worldv2",   # ~80 MB CoreML, COCO AP ~46.5
    "x": "yolov8x-worldv2",   # ~130 MB CoreML, COCO AP ~47.5
}
YOLO_WORLD_DESCRIPTIONS = {
    "s": "YOLO-World V2-S",
    "m": "YOLO-World V2-M",
    "l": "YOLO-World V2-L",
    "x": "YOLO-World V2-X",
}


# ---------------------------------------------------------------------------
# Step 1: YOLO-World Visual Detector
# ---------------------------------------------------------------------------

class YOLOWorldDetectorWrapper(nn.Module):
    """Wraps YOLO-World to accept text embeddings and output split boxes/scores."""

    def __init__(self, world_model):
        super().__init__()
        self.model = world_model

    def forward(self, image, txt_feats):
        self.model.txt_feats = txt_feats
        out = self.model(image)
        pred = out[0]              # [1, 4+NC, 8400]
        boxes = pred[:, :4, :]     # [1, 4, 8400] — cx, cy, w, h
        scores = pred[:, 4:, :]    # [1, NC, 8400] — sigmoid scores
        return boxes, scores


def convert_detector(model_name: str, output_dir: Path, input_size: int = 640):
    """Convert YOLO-World visual detector to CoreML."""
    # Resolve size shorthand
    size_key = model_name.lower()
    if size_key in YOLO_WORLD_MODELS:
        model_name = YOLO_WORLD_MODELS[size_key]
        desc = YOLO_WORLD_DESCRIPTIONS[size_key]
    else:
        desc = model_name
        for k, v in YOLO_WORLD_MODELS.items():
            if model_name == v:
                desc = YOLO_WORLD_DESCRIPTIONS[k]
                break

    print(f"=== Converting Visual Detector ({desc}) ===")

    from ultralytics import YOLO
    model = YOLO(model_name)
    wm = model.model
    wm.eval()

    wrapper = YOLOWorldDetectorWrapper(wm)
    wrapper.eval()

    dummy_img = torch.randn(1, 3, input_size, input_size)
    dummy_txt = torch.randn(1, MAX_CLASSES, 512)

    # Warm up to initialize internal state
    with torch.no_grad():
        _ = wrapper(dummy_img, dummy_txt)
        _ = wrapper(dummy_img, dummy_txt)

    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_img, dummy_txt), check_trace=False)

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="image", shape=(1, 3, input_size, input_size)),
            ct.TensorType(name="txt_feats", shape=(1, MAX_CLASSES, 512)),
        ],
        outputs=[
            ct.TensorType(name="boxes"),
            ct.TensorType(name="scores"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15,
    )

    mlmodel.author = "SAMKit"
    mlmodel.short_description = f"{desc} Visual Detector (open-vocabulary)"
    mlmodel.version = "2.0.0"

    out_path = output_dir / "yoloworld_detector.mlpackage"
    mlmodel.save(str(out_path))
    print(f"  Saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Step 2: CLIP Text Encoder
# ---------------------------------------------------------------------------

class CLIPTextEncoderWrapper(nn.Module):
    """Wraps CLIP text encoder for CoreML-compatible tracing."""

    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, text_tokens):
        x = self.token_embedding(text_tokens)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]
        x = x @ self.text_projection
        return x


def convert_text_encoder(output_dir: Path):
    """Convert CLIP ViT-B/32 text encoder to CoreML."""
    print("\n=== Converting CLIP Text Encoder ===")

    import open_clip

    # Patch MultiheadAttention to avoid _native_multi_head_attention
    # which CoreML does not support
    original_mha = torch.nn.MultiheadAttention.forward
    def patched_mha(self, query, key, value, key_padding_mask=None,
                    need_weights=True, attn_mask=None,
                    average_attn_weights=True, is_causal=False):
        return F.multi_head_attention_forward(
            query, key, value,
            self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
        )
    torch.nn.MultiheadAttention.forward = patched_mha

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model.eval()

    wrapper = CLIPTextEncoderWrapper(clip_model)
    wrapper.eval()

    dummy_tokens = open_clip.tokenize(["placeholder"] * MAX_CLASSES)

    with torch.no_grad():
        _ = wrapper(dummy_tokens)

    print("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_tokens, check_trace=False)

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="text_tokens",
                shape=(MAX_CLASSES, CONTEXT_LENGTH),
                dtype=np.int32,
            )
        ],
        outputs=[
            ct.TensorType(name="text_embeddings"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15,
    )

    mlmodel.author = "SAMKit"
    mlmodel.short_description = "CLIP ViT-B/32 Text Encoder"
    mlmodel.version = "1.0.0"

    out_path = output_dir / "clip_text_encoder.mlpackage"
    mlmodel.save(str(out_path))
    print(f"  Saved to {out_path}")

    torch.nn.MultiheadAttention.forward = original_mha
    return out_path


# ---------------------------------------------------------------------------
# Step 3: CLIP Vocabulary
# ---------------------------------------------------------------------------

def export_vocabulary(output_dir: Path):
    """Export CLIP BPE vocabulary for Swift tokenizer."""
    print("\n=== Exporting CLIP Vocabulary ===")

    from open_clip import tokenizer as clip_tok

    bpe_path = Path(clip_tok.__file__).parent / "bpe_simple_vocab_16e6.txt.gz"
    with gzip.open(str(bpe_path), "rt", encoding="utf-8") as f:
        bpe_data = f.read()

    lines = bpe_data.strip().split("\n")
    merges = [l for l in lines if l and not l.startswith("#")]

    byte_encoder = clip_tok.bytes_to_unicode()
    vocab_list = list(byte_encoder.values())
    vocab_list += [v + "</w>" for v in vocab_list]
    for merge in merges:
        vocab_list.append("".join(merge.split()))
    vocab_list.extend(["<|startoftext|>", "<|endoftext|>"])
    encoder = {v: i for i, v in enumerate(vocab_list)}

    vocab_data = {
        "encoder": encoder,
        "merges": merges,
        "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>",
        "context_length": CONTEXT_LENGTH,
    }

    out_path = output_dir / "clip_vocab.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False)

    print(f"  Saved vocabulary ({len(encoder)} tokens) to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO-World + CLIP to CoreML for open-vocabulary detection"
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
        "--output", type=str, default="../manifests/grounding",
        help="Output directory",
    )
    parser.add_argument(
        "--input-size", type=int, default=640,
        help="Input image size (default: 640)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model name: --model overrides --size, default is L
    if args.model:
        model_name = args.model
    elif args.size:
        model_name = YOLO_WORLD_MODELS[args.size]
    else:
        model_name = YOLO_WORLD_MODELS["l"]
        print("Using default model size: L (use --size s/m/l/x to change)")

    # Convert all components
    det_path = convert_detector(model_name, output_dir, args.input_size)
    text_path = convert_text_encoder(output_dir)
    vocab_path = export_vocabulary(output_dir)

    print(f"\n=== Conversion Complete ===")
    print(f"  Detector:     {det_path}")
    print(f"  Text Encoder: {text_path}")
    print(f"  Vocabulary:   {vocab_path}")
    print(f"\nAdd these files to your Xcode project to enable text-prompted segmentation.")


if __name__ == "__main__":
    main()
