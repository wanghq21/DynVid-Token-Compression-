# DynVid: Adaptive Dynamism-Aware Token Compression for Efficient Video LLMs

## Overview

Video LLMs face efficiency bottlenecks when processing long videos due to excessive visual tokens. DynVid introduces a **dynamism-aware adaptive token compression** framework that operates before LLM decoder, requiring no additional training or model modification.

DynVid Compression consists of:
1. **DySeg** - Dynamic Segmentation via cosine similarity thresholding
2. **ERA** - Effective Rank-based Adaptive budget allocation (Participation Ratio Approximation)
3. **DATS** - Dynamism-Aware Token Selection (Facility Location submodular optimization)
4. **Top-K Soft Fusion** - anchor-protected weighted averaging
5. **LLM Internal Pruning** - attention-based hard pruning at a specified decoder layer

Supported Models: LLaVA-OneVision-7B, LLaVA-Video-7B, Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct



## Installation

```bash
conda create -n dynvid python=3.11
conda activate dynvid

# Install PyTorch (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Install Flash Attention
pip install flash-attn --no-build-isolation

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
dynvid/
├── __init__.py                  # Entry points: dynvid_llava, dynvid_qwen2_5, dynvid_qwen3
├── configuration_dynvid.py      # DynVidConfig dataclass
├── compression_unified.py       # Core compression (DySeg + ERA + DATS + Fusion)
├── modeling_qwen2_5_vl.py       # Qwen2.5-VL integration
├── modeling_qwen3_vl.py         # Qwen3-VL integration
├── modeling_llava_onevision.py  # LLaVA-OneVision / LLaVA-Video integration
└── validate.py                  # EgoSchema dataset submission helper
eval_scripts/
├── llava_ablation.sh            # LLaVA-OneVision evaluation & ablation
├── llava_vid_ablation.sh        # LLaVA-Video evaluation & ablation
├── qwen25_ablation.sh           # Qwen2.5-VL evaluation & ablation
└── qwen3_ablation.sh            # Qwen3-VL evaluation & ablation
lmms-eval/                           # Evaluation framework (from LMMs-Eval)
└── lmms_eval/models/simple/         # The following files are modified to integrate DynVid
    ├── llava_onevision.py               # DynVid wrapper for LLaVA-OneVision
    ├── llava_vid.py                     # DynVid wrapper for LLaVA-Video 
    ├── qwen2_5_vl.py                    # DynVid wrapper for Qwen2.5-VL
    └── qwen3_vl.py                      # DynVid wrapper for Qwen3-VL
```

## Quick Start

```python
# You can override the default parameters (e.g., retention ratio) in the dynvid_qwen2_5, dynvid_qwen3 and dynvid_llava wrapper function.

from dynvid import dynvid_qwen2_5, dynvid_qwen3, dynvid_llava

# Qwen2.5-VL
from transformers import Qwen2_5_VLForConditionalGeneration
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = dynvid_qwen2_5(model)

# Qwen3-VL
from transformers import Qwen3VLForConditionalGeneration
model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
model = dynvid_qwen3(model)

# LLaVA-OneVision / LLaVA-Video
model = dynvid_llava(model)
```

## Evaluation

```bash
# Before running evaluation, configure the HuggingFace cache directory:
export HF_HOME=/path/to/your/huggingface_cache

# LLaVA-OneVision
bash eval_scripts/llava_ablation.sh

# LLaVA-Video
bash eval_scripts/llava_vid_ablation.sh

# Qwen2.5-VL
bash eval_scripts/qwen25_ablation.sh

# Qwen3-VL
bash eval_scripts/qwen3_ablation.sh
```
