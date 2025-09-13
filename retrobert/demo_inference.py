#!/usr/bin/env python3
"""
Demo Inference Script for retroBERT
Uses cehckpoints from outputs/ckpts/fold{1-4} to run inference on test sets
"""

import os
import numpy as np
import argparse
import torch

from data import retroBERTdataset
from model import retroBERT
from inference import test
from config import add_default_args, ARGS_STR
from utils.data_utils import setup_fold_directories, prepare_datasets_with_scaling
from utils.model_utils import set_seed, load_model
from log import check_distribution, initialize_kfold_results, collect_fold_results, print_kfold_table

def load_trained_model(model_path, input_dim, max_seq_length, device):
    """Load a pre-trained model from checkpoint"""
    print(f"Loading model from: {model_path}")
    
    # Create model architecture
    model = retroBERT(input_dim, max_seq_length)
    model.to(device)
    
    # Load the trained weights
    model = load_model(model, model_path)
    
    return model

def main():
    """Main demo inference function"""
    print("=== retroBERT Demo Inference Script ===")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Demo inference with pre-trained models')
    parser = add_default_args(parser)
    args = parser.parse_args(ARGS_STR.split())
    
    # Set random seed for reproducibility
    set_seed(args)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")
    
    # Model configuration
    input_dim = 24
    bert_dim = 768
    
    # Dataset paths
    BASE_DATA_DIR = os.path.join("dataset")
    label_file = os.path.join(BASE_DATA_DIR, "SIratio.xlsx")
    save_dataset_dir = os.path.join(BASE_DATA_DIR, 'fold')
    
    # Pre-trained model base directory
    trained_models_dir = "ckpts"
    
    # Number of folds
    num_folds = 4
    
    # Initialize results tracking 
    results = initialize_kfold_results()
    
    print(f"\nRunning inference on {num_folds} folds...")
    print("=" * 60)
    
    for fold in range(1, num_folds + 1):
        print(f"\n🔍 Processing Fold {fold}")
        print("-" * 60)
        
        # Setup fold directories
        fold_dirs, filename_label_map = setup_fold_directories(fold, save_dataset_dir)
        
        # Create datasets (same as main.py)
        train_dataset = retroBERTdataset(
            data_dir=fold_dirs['train_dir'], 
            label_file=label_file, 
            is_train=True, 
            is_test=False, 
            args=args
        )
        
        valid_dataset = retroBERTdataset(
            data_dir=fold_dirs['valid_dir'], 
            label_file=label_file, 
            is_train=False, 
            is_test=False, 
            args=args
        )
        
        test_dataset = retroBERTdataset(
            data_dir=fold_dirs['test_dir'], 
            label_file=label_file, 
            is_train=False, 
            is_test=True, 
            args=args
        )
        
        # Apply scaling (same as main.py) - use original utils functions
        train_dataset, valid_dataset, train_scaler = prepare_datasets_with_scaling(train_dataset, valid_dataset)
        
        print("Test Dataset Distribution:")
        test_proportions = check_distribution(test_dataset)
        
        # Path to pre-trained models for this fold
        fold_model_dir = os.path.join(trained_models_dir, f"fold{fold}")
        
        if not os.path.exists(fold_model_dir):
            print(f"❌ Model directory not found: {fold_model_dir}")
            print("   Please ensure pre-trained models are saved in outputs/trained/fold{1-4}/")
            continue
        
        # Test
        model_types = ["best_f1"]
        
        for model_type in model_types:
            print(f"\n📋 Testing {model_type} model...")
            
            # Model checkpoint path
            model_path = os.path.join(fold_model_dir, f"checkpoint_{model_type}.pth.tar")
            
            if not os.path.exists(model_path):
                print(f"   ⚠️ Model not found: {model_path}")
                continue
            
            try:
                # Load the pre-trained model
                model = load_trained_model(model_path, input_dim, args.max_seq_length, args.device)
                
                # Run inference
                print(f"   🚀 Running inference with {model_type} model...")
                
                test_accuracy, precision, recall, file_macro_f1 = test(
                    model=model,
                    test_dir=fold_dirs['test_dir'],
                    test_dataset=test_dataset,
                    filename_label_map=filename_label_map,
                    scaler=train_scaler,
                    args=args
                )
                
                # Collect results using updated function
                collect_fold_results(results, test_accuracy, precision, recall, file_macro_f1)
                                
            except Exception as e:
                print(f"   ❌ Error loading/testing {model_type} model: {str(e)}")
                continue
    
    # Print final summary results with SEM in table format
    print("\n" + "=" * 60)
    print("🏁 FINAL INFERENCE RESULTS SUMMARY")
    print("=" * 60)
    
    print_kfold_table(results, num_folds)
    
    print("\n✨ Demo inference completed!")

if __name__ == "__main__":
    main()
