# Pre-trained Models Directory

This directory contains the pre-trained retroBERT model checkpoints for each fold.

## Structure

```
ckpts/
├── fold1/
│   └── checkpoint_best_f1.pth.tar
├── fold2/
│   └── checkpoint_best_f1.pth.tar
├── fold3/
│   └── checkpoint_best_f1.pth.tar
└── fold4/
    └── checkpoint_best_f1.pth.tar
```

## Automatic Download

These models are automatically downloaded from Hugging Face when you run:

```bash
python setup_retrobert.py
```

Or when the CLI detects missing models:

```bash
python run_retrobert.py demo_inference
```

## Manual Download

You can also download only the models:

```bash
python download_models.py
```

## Hugging Face Repository

Models are hosted at: https://huggingface.co/ShinSeungJ/retroBERT
