# retroBERT: Stress Susceptibility Prediction using BERT-based Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

## Overview

retroBERT is a deep learning framework that leverages BERT (Bidirectional Encoder Representations from Transformers) architecture to predict stress susceptibility from behavioral time series data. The model processes sequential behavioral data and classifies subjects as either resilient or susceptible to stress.

## 🚀 Quick Start

### For Replicating Results (Recommended)
```bash
git clone https://github.com/ShinSeungJ/retroBERT.git
cd retroBERT
conda create -n retrobert python=3.9
conda activate retrobert
pip install -e .
python setup_retrobert.py
retrobert demo_inference
```
**Expected time**: ~10-15 minutes total

### For Training from Scratch
```bash
git clone https://github.com/ShinSeungJ/retroBERT.git
cd retroBERT
conda create -n retrobert python=3.9
conda activate retrobert
pip install -e .
python setup_retrobert.py --dataset-only
retrobert main
```
**Expected time**: ~2-4 hours for full k-fold training

### ⚡ Quick Start (if conda already installed)
```bash
git clone https://github.com/ShinSeungJ/retroBERT.git && cd retroBERT && conda create -n retrobert python=3.9 && conda activate retrobert && pip install -e . && python setup_retrobert.py && retrobert demo_inference
```

## System Requirements

### Software Dependencies

- **Python**: 3.9 or higher (recommended), 3.8+ supported
- **PyTorch**: 1.9.0 or higher
- **Transformers**: 4.21.0 or higher (Hugging Face)
- **NumPy**: 1.21.0 or higher
- **Pandas**: 1.3.0 or higher
- **Scikit-learn**: 1.0.0 or higher
- **SciPy**: 1.7.0 or higher
- **OpenPyXL**: 3.0.0 or higher (for Excel file reading)

### Operating Systems

The software has been tested on:
- **Linux**: Ubuntu 20.04 LTS, Ubuntu 22.04 LTS

### Hardware Requirements

- **Minimum**: 8 GB RAM, 2 GB free disk space
- **Recommended**: 16 GB RAM, NVIDIA GPU with 8+ GB VRAM for faster training
- **GPU Support**: CUDA-compatible GPU (optional but recommended for training)
- **Tested on**: NVIDIA GeForce RTX 3060 Ti (8GB VRAM)

### Non-standard Hardware

- NVIDIA GPU with CUDA support (optional, for accelerated training)

## Installation Guide

### Prerequisites: Install Conda

If you don't have conda installed, install it first:

**Miniconda (Recommended - lighter)**
```bash
# Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# macOS
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# Windows: Download from https://docs.conda.io/en/latest/miniconda.html
```

**Or Anaconda (Full distribution)**
- Download from: https://www.anaconda.com/products/distribution

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/ShinSeungJ/retroBERT.git
cd retroBERT

# Create conda environment with Python 3.9
conda create -n retrobert python=3.9
conda activate retrobert

# Install package and dependencies
pip install -e .
```

### Typical Installation Time

- **Standard installation**: 5-10 minutes on a normal desktop computer
- **With GPU support**: 10-15 minutes (includes CUDA toolkit setup)

## Demo

### Automatic Setup (Recommended)

For first-time users, we provide an automatic setup script that downloads pre-trained models and dataset from Hugging Face:

```bash
# Download both models and dataset
python setup_retrobert.py

# Download only models
python setup_retrobert.py --models-only

# Download only dataset  
python setup_retrobert.py --dataset-only
```

### Complete Workflows

#### Workflow 1: Replicate Paper Results
```bash
# Step 1: Clone and install
git clone https://github.com/ShinSeungJ/retroBERT.git
cd retroBERT
conda create -n retrobert python=3.9
conda activate retrobert
pip install -e .

# Step 2: Download pre-trained models and dataset
python setup_retrobert.py

# Step 3: Run inference to get paper results
retrobert demo_inference
```
**Timeline**: Setup (5-10 min) + Download (2-5 min) + Inference (2-5 min) = **~10-20 minutes total**

#### Workflow 2: Train from Scratch
```bash
# Step 1: Clone and install
git clone https://github.com/ShinSeungJ/retroBERT.git
cd retroBERT
conda create -n retrobert python=3.9
conda activate retrobert
pip install -e .

# Step 2: Download dataset only
python setup_retrobert.py --dataset-only

# Step 3: Run full k-fold cross-validation training
retrobert main
```
**Timeline**: Setup (5-10 min) + Download (1-3 min) + Training (2-4 hours) = **~2-4 hours total**

#### Workflow 3: Quick Demo Training
```bash
# Step 1: Clone and install
git clone https://github.com/ShinSeungJ/retroBERT.git
cd retroBERT
conda create -n retrobert python=3.9
conda activate retrobert
pip install -e .

# Step 2: Run demo training (creates demo data automatically)
python run_retrobert.py demo
```
**Timeline**: Setup (5-10 min) + Demo training (5-10 min) = **~10-20 minutes total**

### 🤖 Auto-Download Feature

The CLI automatically detects missing files and offers to download them:

```bash
retrobert demo_inference
# ⚠️  Missing required files:
#    - Pre-trained models (ckpts directory)
#    - Dataset (dataset/fold directory)
# 
# Would you like to download them automatically from Hugging Face? (y/n): y
# 📥 Downloading required files...
# ✅ Download completed!
```

### 🔧 Alternative Execution Methods

After setup, you can run scripts in multiple ways:

```bash
# Using CLI (recommended)
retrobert demo_inference
retrobert main

# Using direct script execution
python retrobert/demo_inference.py
python -m retrobert.main

# After pip installation
retrobert demo_inference
retrobert main
```

### Expected Output

The demo will output:
```
=== retroBERT Demo Inference Script ===
Using device: cuda  # or cpu

🔍 Processing Fold 1
Final Predictions:
Filename  | Prediction | Confidence | Accuracy
pre114.csv: Susceptible|   15.19%   |  62.50%
pre408.csv: Resilient  |   21.12%   |  63.04%
...

============================================================
Metric            Mean ± SEM      Individual Fold Results
============================================================
Accuracy        
Precision       
Recall          
F1              
============================================================
```

### Expected Runtime

- **Demo inference**: 2-5 minutes on a normal desktop computer
- **Full training**: 30-60 minutes per fold (4 folds total) on GPU, 2-4 hours on CPU

## Instructions for Use

### Configuration Options

Key parameters can be modified in `retrobert/config.py` directly:

- `train_epochs`: Number of training epochs (default: 20)
- `learning_rate`: Learning rate (default: 8e-7)
- `train_batch_size`: Training batch size (default: 64)
- `max_seq_length`: Maximum sequence length (default: 64)


**Note**: The `ckpts/` and `dataset/fold/` directories are automatically downloaded from Hugging Face when you run `setup_retrobert.py` or when the CLI detects missing files.

## Troubleshooting

### Common Issues

#### Download Failures
If automatic downloads fail:
```bash
# Try manual download
python setup_retrobert.py

# Or download components separately
python download_models.py
python download_dataset.py
```

#### Missing Dependencies
If you get import errors:
```bash
# Reinstall package and dependencies
pip install -e .

# Or install specific missing packages
pip install huggingface-hub transformers torch
```

#### CUDA/GPU Issues
If you encounter GPU-related errors:
```bash
# The code automatically falls back to CPU
# Check if CUDA is available:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Algorithm Description

The complete algorithmic details and pseudocode are provided in the **Methods** section of the manuscript.

## Support

For questions and support, please:
1. Check the [Issues](https://github.com/ShinSeungJ/retroBERT/issues) page
2. Create a new issue if your question hasn't been addressed
3. Contact: [shin93754@kaist.ackr]
