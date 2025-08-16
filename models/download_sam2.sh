#!/bin/bash

# SAM2 Model Download Script

echo "SAM2.1 Model Downloader"
echo "======================="

# Create model directory
mkdir -p checkpoints

# SAM2.1 model URLs (from Meta's official release)
SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

# Model options
echo "Select SAM2.1 model to download:"
echo "1) SAM2.1 Tiny (39M params)"
echo "2) SAM2.1 Small (46M params)"
echo "3) SAM2.1 Base+ (81M params)"
echo "4) SAM2.1 Large (224M params)"
echo "5) All models"

read -p "Enter choice (1-5): " choice

download_model() {
    local model_name=$1
    local filename=$2
    local url="${SAM2_BASE_URL}/${filename}"
    
    echo "Downloading ${model_name}..."
    if command -v wget &> /dev/null; then
        wget -O "checkpoints/${filename}" "${url}"
    elif command -v curl &> /dev/null; then
        curl -L -o "checkpoints/${filename}" "${url}"
    else
        echo "Error: Neither wget nor curl is installed"
        exit 1
    fi
    echo "✓ ${model_name} downloaded"
}

case $choice in
    1)
        download_model "SAM2.1 Tiny" "sam2.1_hiera_tiny.pt"
        ;;
    2)
        download_model "SAM2.1 Small" "sam2.1_hiera_small.pt"
        ;;
    3)
        download_model "SAM2.1 Base+" "sam2.1_hiera_base_plus.pt"
        ;;
    4)
        download_model "SAM2.1 Large" "sam2.1_hiera_large.pt"
        ;;
    5)
        download_model "SAM2.1 Tiny" "sam2.1_hiera_tiny.pt"
        download_model "SAM2.1 Small" "sam2.1_hiera_small.pt"
        download_model "SAM2.1 Base+" "sam2.1_hiera_base_plus.pt"
        download_model "SAM2.1 Large" "sam2.1_hiera_large.pt"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Download complete! Models saved in checkpoints/"
echo ""
echo "Next steps:"
echo "1. Convert to Core ML (iOS): python converters/convert_to_coreml.py --model sam2_base --checkpoint checkpoints/sam2.1_hiera_base_plus.pt"
echo "2. Run inference: python test_sam2.py --checkpoint checkpoints/sam2.1_hiera_base_plus.pt --image your_image.jpg"