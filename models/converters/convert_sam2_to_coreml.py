#!/usr/bin/env python3
"""
Convert SAM2 models to Core ML format for iOS
Specialized for SAM2.1 architecture
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
from PIL import Image

# Import SAM2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Installing SAM2...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/sam2.git"])
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2EncoderWrapper(nn.Module):
    """Wrapper for SAM2 image encoder to handle Core ML conversion"""
    
    def __init__(self, model, image_size=1024):
        super().__init__()
        self.image_encoder = model.image_encoder
        self.image_size = image_size
    
    def forward(self, x):
        # x shape: (1, 3, 1024, 1024)
        # SAM2 expects the image to be preprocessed
        features = self.image_encoder(x)
        
        # Handle different feature formats
        if isinstance(features, (list, tuple)):
            # Return the main feature map
            return features[0]
        return features


class SAM2DecoderWrapper(nn.Module):
    """Wrapper for SAM2 mask decoder to handle Core ML conversion"""
    
    def __init__(self, model, image_size=1024):
        super().__init__()
        self.mask_decoder = model.sam_mask_decoder
        self.prompt_encoder = model.sam_prompt_encoder
        self.image_size = image_size
        
    @torch.no_grad()
    def forward(self, image_embeddings, point_coords, point_labels):
        """
        Simplified decoder for Core ML
        Args:
            image_embeddings: (1, C, H, W) - from encoder
            point_coords: (1, N, 2) - xy coordinates
            point_labels: (1, N) - 0 or 1 labels
        """
        batch_size = 1
        
        # Encode prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        
        # Run decoder
        low_res_masks, iou_predictions, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=None,
        )
        
        return low_res_masks, iou_predictions


def load_sam2_model(checkpoint_path: str, model_cfg: str, device: str = "cpu"):
    """Load SAM2 model"""
    print(f"Loading SAM2 model from {checkpoint_path}...")
    
    # Build model
    sam2 = build_sam2(model_cfg, checkpoint_path, device=device)
    sam2.eval()
    
    return sam2


def trace_encoder(model, image_size: int = 1024):
    """Trace SAM2 encoder for Core ML conversion"""
    print("Tracing encoder...")
    
    wrapped_encoder = SAM2EncoderWrapper(model, image_size)
    wrapped_encoder.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, image_size, image_size)
    
    # Trace the model
    with torch.no_grad():
        traced_encoder = torch.jit.trace(wrapped_encoder, example_input)
    
    return traced_encoder


def convert_encoder_to_coreml(traced_encoder, output_path: Path, image_size: int = 1024):
    """Convert traced encoder to Core ML"""
    print("Converting encoder to Core ML...")
    
    # Define input
    example_input = torch.randn(1, 3, image_size, image_size)
    
    # Convert to Core ML
    mlmodel = ct.convert(
        traced_encoder,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, image_size, image_size),
                scale=1.0/255.0,
                bias=[0, 0, 0],
                color_layout=ct.colorlayout.RGB
            )
        ],
        outputs=[
            ct.TensorType(name="image_embeddings")
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram"
    )
    
    # Add metadata
    mlmodel.author = "SAMKit"
    mlmodel.short_description = "SAM2 Image Encoder"
    mlmodel.version = "2.1"
    
    # Save model
    mlmodel.save(str(output_path))
    print(f"Encoder saved to {output_path}")
    
    return output_path


def trace_decoder(model, image_size: int = 1024):
    """Trace SAM2 decoder for Core ML conversion"""
    print("Tracing decoder...")
    
    wrapped_decoder = SAM2DecoderWrapper(model, image_size)
    wrapped_decoder.eval()
    
    # Create example inputs
    # Get actual embedding size from model
    with torch.no_grad():
        test_image = torch.randn(1, 3, image_size, image_size)
        test_features = model.image_encoder(test_image)
        if isinstance(test_features, (list, tuple)):
            test_features = test_features[0]
        embed_dim = test_features.shape[1]
        embed_size = test_features.shape[2]
    
    image_embeddings = torch.randn(1, embed_dim, embed_size, embed_size)
    point_coords = torch.randn(1, 5, 2) * image_size
    point_labels = torch.randint(0, 2, (1, 5)).float()
    
    # Trace the model
    with torch.no_grad():
        traced_decoder = torch.jit.trace(
            wrapped_decoder,
            (image_embeddings, point_coords, point_labels)
        )
    
    return traced_decoder, embed_dim, embed_size


def convert_decoder_to_coreml(traced_decoder, output_path: Path, 
                             embed_dim: int, embed_size: int, image_size: int = 1024):
    """Convert traced decoder to Core ML"""
    print("Converting decoder to Core ML...")
    
    # Create example inputs for shape inference
    image_embeddings = torch.randn(1, embed_dim, embed_size, embed_size)
    point_coords = torch.randn(1, 5, 2) * image_size
    point_labels = torch.randint(0, 2, (1, 5)).float()
    
    # Convert to Core ML with flexible shapes
    mlmodel = ct.convert(
        traced_decoder,
        inputs=[
            ct.TensorType(
                name="image_embeddings",
                shape=(1, embed_dim, embed_size, embed_size)
            ),
            ct.TensorType(
                name="point_coords",
                shape=(1, ct.RangeDim(1, 10), 2)  # Variable number of points
            ),
            ct.TensorType(
                name="point_labels",
                shape=(1, ct.RangeDim(1, 10))
            )
        ],
        outputs=[
            ct.TensorType(name="masks"),
            ct.TensorType(name="iou_predictions")
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram"
    )
    
    # Add metadata
    mlmodel.author = "SAMKit"
    mlmodel.short_description = "SAM2 Mask Decoder"
    mlmodel.version = "2.1"
    
    # Save model
    mlmodel.save(str(output_path))
    print(f"Decoder saved to {output_path}")
    
    return output_path


def compute_model_hash(model_path: Path) -> str:
    """Compute SHA256 hash of model file"""
    sha256 = hashlib.sha256()
    with open(model_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_manifest(model_name: str, encoder_path: Path, decoder_path: Path,
                   output_dir: Path, image_size: int = 1024):
    """Create model manifest JSON"""
    
    manifest = {
        "name": model_name,
        "version": "2.1",
        "input_size": image_size,
        "encoder": {
            "type": "coreml",
            "path": encoder_path.name,
            "precision": "fp16",
            "hash": compute_model_hash(encoder_path)
        },
        "decoder": {
            "type": "coreml",
            "path": decoder_path.name,
            "precision": "fp16",
            "hash": compute_model_hash(decoder_path)
        },
        "normalize": {
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375]
        }
    }
    
    manifest_path = output_dir / f"{model_name}_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Created manifest at {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Convert SAM2 models to Core ML")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to SAM2 checkpoint")
    parser.add_argument("--model-type", type=str, required=True,
                       choices=["sam2_tiny", "sam2_small", "sam2_base_plus", "sam2_large"],
                       help="Model type")
    parser.add_argument("--output", type=str, default="../manifests",
                       help="Output directory")
    parser.add_argument("--image-size", type=int, default=1024,
                       help="Input image size")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda", "mps"],
                       help="Device for tracing")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map model type to config
    model_configs = {
        "sam2_tiny": "sam2_hiera_t",
        "sam2_small": "sam2_hiera_s",
        "sam2_base_plus": "sam2_hiera_b+",
        "sam2_large": "sam2_hiera_l"
    }
    
    model_cfg = model_configs[args.model_type]
    
    # Load model
    model = load_sam2_model(args.checkpoint, model_cfg, args.device)
    
    # Trace encoder
    traced_encoder = trace_encoder(model, args.image_size)
    
    # Convert encoder
    encoder_path = output_dir / f"{args.model_type}_encoder.mlpackage"
    convert_encoder_to_coreml(traced_encoder, encoder_path, args.image_size)
    
    # Trace decoder
    traced_decoder, embed_dim, embed_size = trace_decoder(model, args.image_size)
    
    # Convert decoder
    decoder_path = output_dir / f"{args.model_type}_decoder.mlpackage"
    convert_decoder_to_coreml(traced_decoder, decoder_path, embed_dim, embed_size, args.image_size)
    
    # Create manifest
    create_manifest(args.model_type, encoder_path, decoder_path, output_dir, args.image_size)
    
    print(f"\n✅ Conversion complete!")
    print(f"Models saved to {output_dir}")
    print(f"\nTo use in iOS:")
    print(f"1. Copy {encoder_path.name} and {decoder_path.name} to your Xcode project")
    print(f"2. Use with SAMKit:")
    print(f"   let model = SamModelRef(")
    print(f'       encoderURL: Bundle.main.url(forResource: "{args.model_type}_encoder", withExtension: "mlmodelc")!,')
    print(f'       decoderURL: Bundle.main.url(forResource: "{args.model_type}_decoder", withExtension: "mlmodelc")!,')
    print(f"       modelType: .sam2_{args.model_type.split('_')[1]}")
    print(f"   )")


if __name__ == "__main__":
    main()