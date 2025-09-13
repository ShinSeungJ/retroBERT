# K-Fold Dataset Directory

This directory contains the k-fold cross-validation splits of the dataset.

## Structure

```
fold/
├── train_fold_1/
├── train_fold_2/
├── train_fold_3/
├── train_fold_4/
├── valid_fold_1/
├── valid_fold_2/
├── valid_fold_3/
├── valid_fold_4/
├── test_fold_1/
├── test_fold_2/
├── test_fold_3/
└── test_fold_4/
```

Each fold directory contains CSV files with behavioral time series data.

## Automatic Download

The dataset is automatically downloaded from Hugging Face when you run:

```bash
python setup_retrobert.py
```

Or when the CLI detects missing dataset:

```bash
python run_retrobert.py main
```

## Manual Download

You can also download only the dataset:

```bash
python download_dataset.py
```

## Hugging Face Repository

Dataset is hosted at: https://huggingface.co/datasets/ShinSeungJ/retroBERT-dataset
