#!/usr/bin/env python3
"""
Convert SAM 2.1 and MobileSAM models to Core ML format for iOS
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from PIL import Image


class SAMEncoderWrapper(nn.Module):
    """Wrapper for SAM image encoder to handle Core ML conversion"""
    
    def __init__(self, encoder, image_size=1024):
        super().__init__()
        self.encoder = encoder
        self.image_size = image_size
    
    def forward(self, x):
        # x shape: (1, 3, 1024, 1024)
        return self.encoder(x)


class SAMDecoderWrapper(nn.Module):
    """Wrapper for SAM mask decoder to handle Core ML conversion"""
    
    def __init__(self, decoder, image_size=1024):
        super().__init__()
        self.decoder = decoder
        self.image_size = image_size
        self.embed_dim = decoder.embed_dim if hasattr(decoder, 'embed_dim') else 256
        
    def forward(self, image_embeddings, point_coords, point_labels, mask_input, has_mask_input):
        """
        Args:
            image_embeddings: (1, embed_dim, H/16, W/16)
            point_coords: (1, N, 2) - xy coordinates
            point_labels: (1, N) - 0 or 1 labels
            mask_input: (1, 1, 256, 256) - previous mask
            has_mask_input: (1,) - whether mask_input is valid
        """
        # Prepare inputs for decoder
        sparse_embeddings = self._embed_points(point_coords, point_labels)
        dense_embeddings = self._embed_masks(mask_input) if has_mask_input[0] > 0 else None
        
        # Run decoder
        masks, iou_predictions = self.decoder(
            image_embeddings=image_embeddings,
            image_pe=self.decoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True
        )
        
        return masks, iou_predictions
    
    def _embed_points(self, coords, labels):
        """Embed point prompts"""
        # Simplified point embedding
        # In real implementation, this would use the prompt encoder
        return torch.randn(1, coords.shape[1], self.embed_dim)
    
    def _embed_masks(self, masks):
        """Embed mask prompts"""
        # Simplified mask embedding
        return torch.randn(1, self.embed_dim, 64, 64)


def load_sam_model(model_type: str, checkpoint_path: Optional[str] = None) -> Tuple[nn.Module, nn.Module]:
    """Load SAM or MobileSAM model"""
    
    if model_type.startswith("sam2"):
        # Load SAM 2.1 model
        from segment_anything import sam_model_registry
        
        if model_type == "sam2_tiny":
            model = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        elif model_type == "sam2_small":
            model = sam_model_registry["vit_s"](checkpoint=checkpoint_path)
        elif model_type == "sam2_base":
            model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        elif model_type == "sam2_large":
            model = sam_model_registry["vit_l"](checkpoint=checkpoint_path)
        elif model_type == "sam2_base_plus":
            model = sam_model_registry["vit_b_plus"](checkpoint=checkpoint_path)
        else:
            raise ValueError(f"Unknown SAM model type: {model_type}")
            
    elif model_type == "mobile_sam":
        # Load MobileSAM model
        from mobile_sam import sam_model_registry
        model = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    return model.image_encoder, model.mask_decoder


def convert_encoder(encoder: nn.Module, 
                   model_name: str,
                   image_size: int = 1024,
                   output_dir: Path = Path(".")):
    """Convert SAM encoder to Core ML"""
    
    print(f"Converting {model_name} encoder to Core ML...")
    
    # Wrap encoder
    wrapped_encoder = SAMEncoderWrapper(encoder, image_size)
    wrapped_encoder.eval()
    
    # Prepare sample input
    example_input = torch.randn(1, 3, image_size, image_size)
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped_encoder, example_input)
    
    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
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
        minimum_deployment_target=ct.target.iOS15
    )
    
    # Add metadata
    mlmodel.author = "SAMKit"
    mlmodel.short_description = f"{model_name} Image Encoder"
    mlmodel.version = "1.0.0"
    
    # Save model
    output_path = output_dir / f"{model_name}_encoder.mlpackage"
    mlmodel.save(str(output_path))
    print(f"Saved encoder to {output_path}")
    
    return output_path


def convert_decoder(decoder: nn.Module,
                   model_name: str,
                   embed_dim: int = 256,
                   image_size: int = 1024,
                   output_dir: Path = Path(".")):
    """Convert SAM decoder to Core ML"""
    
    print(f"Converting {model_name} decoder to Core ML...")
    
    # Wrap decoder
    wrapped_decoder = SAMDecoderWrapper(decoder, image_size)
    wrapped_decoder.eval()
    
    # Prepare sample inputs
    embed_size = image_size // 16  # Typical for ViT-based models
    image_embeddings = torch.randn(1, embed_dim, embed_size, embed_size)
    point_coords = torch.randn(1, 5, 2) * image_size  # 5 points
    point_labels = torch.randint(0, 2, (1, 5)).float()  # Binary labels
    mask_input = torch.randn(1, 1, 256, 256)
    has_mask_input = torch.tensor([0.0])  # No mask input initially
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapped_decoder,
            (image_embeddings, point_coords, point_labels, mask_input, has_mask_input)
        )
    
    # Define flexible input shapes for Core ML
    ct_inputs = [
        ct.TensorType(
            name="image_embeddings",
            shape=(1, embed_dim, embed_size, embed_size)
        ),
        ct.TensorType(
            name="point_coords",
            shape=(1, ct.RangeDim(0, 10), 2)  # Variable number of points
        ),
        ct.TensorType(
            name="point_labels",
            shape=(1, ct.RangeDim(0, 10))
        ),
        ct.TensorType(
            name="mask_input",
            shape=(1, 1, 256, 256)
        ),
        ct.TensorType(
            name="has_mask_input",
            shape=(1,)
        )
    ]
    
    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
        inputs=ct_inputs,
        outputs=[
            ct.TensorType(name="masks"),
            ct.TensorType(name="iou_predictions")
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15
    )
    
    # Add metadata
    mlmodel.author = "SAMKit"
    mlmodel.short_description = f"{model_name} Mask Decoder"
    mlmodel.version = "1.0.0"
    
    # Save model
    output_path = output_dir / f"{model_name}_decoder.mlpackage"
    mlmodel.save(str(output_path))
    print(f"Saved decoder to {output_path}")
    
    return output_path


def compute_model_hash(model_path: Path) -> str:
    """Compute SHA256 hash of model file"""
    sha256 = hashlib.sha256()
    with open(model_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_manifest(model_name: str,
                   encoder_path: Path,
                   decoder_path: Path,
                   output_dir: Path,
                   image_size: int = 1024):
    """Create model manifest JSON"""
    
    manifest = {
        "name": model_name,
        "version": "1.0.0",
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
    parser = argparse.ArgumentParser(description="Convert SAM models to Core ML")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["sam2_tiny", "sam2_small", "sam2_base", "sam2_large", 
                "sam2_base_plus", "mobile_sam"],
        help="Model type to convert"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../manifests",
        help="Output directory"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Input image size"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    encoder, decoder = load_sam_model(args.model, args.checkpoint)
    
    # Convert encoder and decoder
    encoder_path = convert_encoder(
        encoder, 
        args.model,
        args.image_size,
        output_dir
    )
    
    # Determine embedding dimension based on model
    embed_dims = {
        "sam2_tiny": 96,
        "sam2_small": 384, 
        "sam2_base": 768,
        "sam2_large": 1024,
        "sam2_base_plus": 768,
        "mobile_sam": 256
    }
    embed_dim = embed_dims.get(args.model, 256)
    
    decoder_path = convert_decoder(
        decoder,
        args.model,
        embed_dim,
        args.image_size,
        output_dir
    )
    
    # Create manifest
    create_manifest(
        args.model,
        encoder_path,
        decoder_path,
        output_dir,
        args.image_size
    )
    
    print(f"\nConversion complete! Models saved to {output_dir}")


if __name__ == "__main__":
    main()