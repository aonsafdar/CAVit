# CA-ViT: Channel-Aware Vision Transformers for Dynamic Feature Fusion

This repository contains the official PyTorch implementation of **CA-ViT**, a lightweight Vision Transformer that replaces static MLP-based channel mixing with dynamic attention across feature channels.

> 📄 **Paper**: _CA-ViT: Channel-Aware Vision Transformers for Dynamic Feature Fusion_  
> 📍 **Venue**: T4V Workshop @ CVPR 2025  
> 🔗 [Project Page / Paper Link (Coming Soon)]()

## 🧠 Overview

CA-ViT is a minimal yet powerful extension to standard Vision Transformers (ViTs). It introduces a second attention mechanism per Transformer block that operates over feature channels via a simple dimension swap. This allows:

- Dynamic, data-dependent feature fusion across channels  
- Enhanced global context modeling  
- Reduced parameter count and FLOPs, with improved accuracy  

<p align="center">
  <img src="figures/arch_diagram.png" width="600"/>
</p>

## 🚀 Features

- Dual-attention ViT block (spatial + channel attention)
- Drop-in replacement for MLP in ViT blocks
- Clean implementation using [timm](https://github.com/huggingface/pytorch-image-models)
- Visualization utilities for token attention maps
- Lightweight and reproducible experiments on 5 benchmark datasets

---

## 📦 Setup

```bash
# Create environment (optional)
conda create -n cavit python=3.10
conda activate cavit

# Install dependencies
pip install -r requirements.txt
