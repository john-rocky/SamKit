#!/usr/bin/env python3
"""
Test SAM2 model with interactive prompts
"""

import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Try to import SAM2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("SAM2 not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/sam2.git"])
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

def load_image(image_path):
    """Load and prepare image"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show_mask(mask, ax, random_color=False):
    """Display mask on plot"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    """Display points on plot"""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', 
              marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', 
              marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """Display bounding box on plot"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', 
                               facecolor=(0, 0, 0, 0), lw=2))

def interactive_segmentation(predictor, image):
    """Run interactive segmentation"""
    print("\n=== Interactive Segmentation ===")
    print("Click on the image to add points:")
    print("- Left click: positive point (include)")
    print("- Right click: negative point (exclude)")
    print("- Middle click or 'q': finish")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Click to add points")
    axes[0].axis('on')
    
    # Result
    axes[1].imshow(image)
    axes[1].set_title("Segmentation result")
    axes[1].axis('off')
    
    points = []
    labels = []
    
    def onclick(event):
        if event.inaxes != axes[0]:
            return
        
        if event.button == 1:  # Left click - positive
            points.append([event.xdata, event.ydata])
            labels.append(1)
            axes[0].plot(event.xdata, event.ydata, 'g*', markersize=15)
        elif event.button == 3:  # Right click - negative
            points.append([event.xdata, event.ydata])
            labels.append(0)
            axes[0].plot(event.xdata, event.ydata, 'r*', markersize=15)
        elif event.button == 2:  # Middle click - finish
            plt.close()
            return
        
        # Update segmentation
        if len(points) > 0:
            input_points = np.array(points)
            input_labels = np.array(labels)
            
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            
            # Show best mask
            best_idx = np.argmax(scores)
            axes[1].clear()
            axes[1].imshow(image)
            show_mask(masks[best_idx], axes[1])
            show_points(input_points, input_labels, axes[1])
            axes[1].set_title(f"Segmentation (score: {scores[best_idx]:.3f})")
            axes[1].axis('off')
        
        plt.draw()
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

def batch_segmentation(predictor, image, mode='grid'):
    """Run batch segmentation with automatic prompts"""
    print("\n=== Batch Segmentation ===")
    
    if mode == 'grid':
        # Generate grid of points
        h, w = image.shape[:2]
        grid_size = 32
        points = []
        for y in range(grid_size, h, grid_size):
            for x in range(grid_size, w, grid_size):
                points.append([x, y])
        
        input_points = np.array(points)
        input_labels = np.ones(len(points))  # All positive
        
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
        
        # Display results
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.scatter(input_points[:, 0], input_points[:, 1], c='red', s=10)
        plt.title("Input Points")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        show_mask(masks[0], plt.gca())
        plt.title(f"Segmentation (score: {scores[0]:.3f})")
        plt.axis('off')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Test SAM2 model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to SAM2 checkpoint")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to input image")
    parser.add_argument("--model-cfg", type=str, default="sam2_hiera_b+",
                       choices=["sam2_hiera_t", "sam2_hiera_s", 
                               "sam2_hiera_b+", "sam2_hiera_l"],
                       help="Model configuration")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu", "mps"],
                       help="Device to use")
    parser.add_argument("--mode", type=str, default="interactive",
                       choices=["interactive", "batch", "auto"],
                       help="Segmentation mode")
    parser.add_argument("--points", type=str,
                       help="Manual points as 'x1,y1,label1;x2,y2,label2'")
    parser.add_argument("--box", type=str,
                       help="Bounding box as 'x1,y1,x2,y2'")
    
    args = parser.parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = "cpu"
    
    print(f"Using device: {args.device}")
    
    # Load model
    print(f"Loading SAM2 model from {args.checkpoint}...")
    sam2 = build_sam2(args.model_cfg, args.checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(sam2)
    
    # Load image
    print(f"Loading image from {args.image}...")
    image = load_image(args.image)
    print(f"Image shape: {image.shape}")
    
    # Set image
    predictor.set_image(image)
    
    if args.mode == "interactive":
        interactive_segmentation(predictor, image)
    elif args.mode == "batch":
        batch_segmentation(predictor, image)
    elif args.mode == "auto":
        # Automatic mask generation
        print("Generating automatic masks...")
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        
        mask_generator = SAM2AutomaticMaskGenerator(sam2)
        masks = mask_generator.generate(image)
        
        print(f"Generated {len(masks)} masks")
        
        # Sort by area
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Display top masks
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (ax, mask_data) in enumerate(zip(axes, masks[:6])):
            ax.imshow(image)
            mask = mask_data['segmentation']
            show_mask(mask, ax, random_color=True)
            ax.set_title(f"Mask {idx+1} (area: {mask_data['area']})")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Manual points/box if provided
    if args.points or args.box:
        input_points = None
        input_labels = None
        input_box = None
        
        if args.points:
            points_data = []
            labels_data = []
            for point_str in args.points.split(';'):
                x, y, label = point_str.split(',')
                points_data.append([float(x), float(y)])
                labels_data.append(int(label))
            input_points = np.array(points_data)
            input_labels = np.array(labels_data)
        
        if args.box:
            coords = [float(x) for x in args.box.split(',')]
            input_box = np.array(coords)
        
        # Predict
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box,
            multimask_output=True,
        )
        
        # Display results
        fig, axes = plt.subplots(1, min(3, len(masks)), figsize=(15, 5))
        if len(masks) == 1:
            axes = [axes]
        
        for idx, (ax, mask, score) in enumerate(zip(axes, masks, scores)):
            ax.imshow(image)
            show_mask(mask, ax)
            if input_points is not None:
                show_points(input_points, input_labels, ax)
            if input_box is not None:
                show_box(input_box, ax)
            ax.set_title(f"Mask {idx+1} (score: {score:.3f})")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()