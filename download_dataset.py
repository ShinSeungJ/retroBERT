#!/usr/bin/env python3
"""
Download dataset from Hugging Face Hub
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("❌ huggingface_hub not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download

def download_dataset(target_dir="dataset"):
    """
    Download dataset from Hugging Face Hub
    
    Args:
        target_dir (str): Directory to save the dataset (default: "dataset")
    """
    print("🤗 Downloading dataset from Hugging Face...")
    print("=" * 60)
    
    repo_id = "ShinSeungJ/retroBERT-dataset"
    
    try:
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Download the entire dataset
        print(f"📥 Downloading dataset to: {target_dir}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        
        print("✅ Dataset downloaded successfully!")
        print(f"   Location: {os.path.abspath(target_dir)}")
        
        # Verify download
        fold_dir = os.path.join(target_dir, "fold")
        if os.path.exists(fold_dir):
            print(f"   ✓ fold directory found")
            
            # Check for expected fold structure
            expected_folds = ["train_fold_1", "train_fold_2", "train_fold_3", "train_fold_4",
                            "valid_fold_1", "valid_fold_2", "valid_fold_3", "valid_fold_4",
                            "test_fold_1", "test_fold_2", "test_fold_3", "test_fold_4"]
            
            found_folds = []
            for fold in expected_folds:
                fold_path = os.path.join(fold_dir, fold)
                if os.path.exists(fold_path):
                    found_folds.append(fold)
                    print(f"   ✓ {fold}")
                else:
                    print(f"   ⚠️  {fold} not found")
            
            print(f"   Found {len(found_folds)}/{len(expected_folds)} fold directories")
        else:
            print(f"   ❌ fold directory not found in {target_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        print("   Please check your internet connection and try again.")
        return False

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download retroBERT dataset")
    parser.add_argument("--output-dir", type=str, default="dataset",
                       help="Directory to save dataset (default: dataset)")
    
    args = parser.parse_args()
    
    success = download_dataset(args.output_dir)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
