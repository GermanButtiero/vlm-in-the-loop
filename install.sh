#!/bin/bash

ENV_NAME="ovis"
PYTHON_VERSION="3.10"

echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Ensure conda is initialized
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio

echo "Installing main packages..."
pip install \
  transformers==4.46.2 \
  numpy==1.25.0 \
  pillow==10.3.0 \
  matplotlib==3.6.0 \
  pycocotools==2.0.5 \
  opencv-python==4.7.0.72 \
  mongoengine==0.29.1 \
  fiftyone==1.4.0 \
  fiftyone-brain==0.20.1 \
  fiftyone-db==1.1.7

echo "Installing flash-attn after torch..."
pip install flash-attn==2.7.1.post4 --no-build-isolation -v

echo "Environment setup complete."