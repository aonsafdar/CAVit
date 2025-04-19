# CAViT: Channel-Aware Vision Transformers for Dynamic Feature Fusion

This repository contains the official PyTorch implementation of **CAViT**


## 📦 Setup

```bash
# Create virtual environment
python -m venv cavit-env
source cavit-env/bin/activate  # or .\cavit-env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run Script
python main.py --dataset pneumoniamnist \
               --epochs 100 \
               --batch_size 128 \
               --image_size 224 \
               --patch_size 16 
