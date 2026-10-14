# Qwen-Image-2.1 — Open-Weight Image Generation

Running and experimenting with **Qwen-Image-2.1**, an open-weight image-generation model, using PyTorch, Hugging Face Diffusers, and NVIDIA H100 GPU acceleration.

## Overview

This repository documents hands-on experimentation with open-weight generative AI models, focusing on model deployment, GPU inference, image generation, and future optimization techniques.

The initial implementation loads pretrained Qwen-Image-2.1 weights and performs text-to-image inference through a Jupyter notebook.

## Hardware & Environment

- **GPU:** NVIDIA H100 80GB HBM3
- **Compute:** Nibi HPC Cluster (Digital Research Alliance of Canada)
- **Python:** 3.11.5
- **PyTorch:** 2.14.1
- **Transformers:** 5.19.0
- **Diffusers:** 0.41.0.dev0
- **Precision:** BF16

## Getting Started

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install torch torchvision transformers accelerate safetensors pillow huggingface_hub
pip install git+https://github.com/huggingface/diffusers.git
```

### 3. Load the model

```python
import torch
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1",
    dtype=torch.bfloat16,
).to("cuda")
```

### 4. Generate an image

```python
prompt = "A futuristic city at night, cinematic lighting, highly detailed"

image = pipe(
    prompt=prompt,
    width=1024,
    height=1024,
    num_inference_steps=30,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image.save("output.png")
```

## Initial Results

- Successfully loaded pretrained model weights onto an NVIDIA H100.
- Observed approximately **30.9 GB of GPU memory usage** after model loading.
- Successfully generated 1024 × 1024 images using text prompts.
- Ran inference locally without a hosted image-generation API.

## Future Experiments

- Image-to-image editing and reference-guided generation
- GPU memory and inference latency benchmarking
- Mixed-precision and quantized inference
- LoRA-based domain adaptation
- Image quality and prompt adherence evaluation

## References

- [Qwen-Image-2.1 — Hugging Face](https://huggingface.co/Qwen/Qwen-Image-2.1)
- [Qwen-Image-2.1 — GitHub](https://github.com/QwenLM/Qwen-Image-2.1)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)

## Acknowledgments

Model developed by the Qwen team. Inference implemented using Hugging Face Diffusers and PyTorch.
