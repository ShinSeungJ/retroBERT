#!/usr/bin/env python3
"""
Setup script for retroBERT - Downloads models and dataset from Hugging Face
"""

import os
import sys
from pathlib import Path

def check_and_install_huggingface_hub():
    """Check if huggingface_hub is installed, install if not"""
    try:
        import huggingface_hub
        return True
    except ImportError:
        print("📦 Installing huggingface_hub...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install huggingface_hub")
            return False

def download_models(target_dir="ckpts"):
    """Download pre-trained models from Hugging Face Hub"""
    from huggingface_hub import snapshot_download
    
    print("🤗 Downloading pre-trained models...")
    print("-" * 40)
    
    repo_id = "ShinSeungJ/retroBERT"
    
    try:
        os.makedirs(target_dir, exist_ok=True)
        
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=target_dir,
            allow_patterns="ckpts/**",
            local_dir_use_symlinks=False
        )
        
        # Flatten structure if needed
        nested_ckpts = os.path.join(target_dir, "ckpts")
        if os.path.exists(nested_ckpts):
            import shutil
            for item in os.listdir(nested_ckpts):
                src = os.path.join(nested_ckpts, item)
                dst = os.path.join(target_dir, item)
                if os.path.exists(dst):
                    shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
                shutil.move(src, dst)
            os.rmdir(nested_ckpts)
        
        print("✅ Models downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading models: {str(e)}")
        return False

def download_dataset(target_dir="dataset"):
    """Download dataset from Hugging Face Hub"""
    from huggingface_hub import snapshot_download
    
    print("🤗 Downloading dataset...")
    print("-" * 40)
    
    repo_id = "ShinSeungJ/retroBERT-dataset"
    
    try:
        os.makedirs(target_dir, exist_ok=True)
        
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        
        print("✅ Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        return False

def verify_setup():
    """Verify that all required files are in place"""
    print("\n🔍 Verifying setup...")
    print("-" * 40)
    
    success = True
    
    # Check models
    ckpts_dir = "ckpts"
    if os.path.exists(ckpts_dir):
        print("✓ Models directory found")
        for fold in ["fold1", "fold2", "fold3", "fold4"]:
            model_path = os.path.join(ckpts_dir, fold, "checkpoint_best_f1.pth.tar")
            if os.path.exists(model_path):
                print(f"  ✓ {fold}/checkpoint_best_f1.pth.tar")
            else:
                print(f"  ❌ {fold}/checkpoint_best_f1.pth.tar missing")
                success = False
    else:
        print("❌ Models directory not found")
        success = False
    
    # Check dataset
    dataset_dir = "dataset/fold"
    if os.path.exists(dataset_dir):
        print("✓ Dataset directory found")
        fold_count = 0
        for item in os.listdir(dataset_dir):
            if item.startswith(("train_fold_", "valid_fold_", "test_fold_")):
                fold_count += 1
        print(f"  ✓ Found {fold_count} fold directories")
        if fold_count < 12:  # 4 train + 4 valid + 4 test = 12
            print("  ⚠️  Some fold directories may be missing")
    else:
        print("❌ Dataset fold directory not found")
        success = False
    
    return success

def main():
    """Main setup function"""
    print("🚀 retroBERT Setup Script")
    print("=" * 60)
    print("This script will download pre-trained models and dataset from Hugging Face")
    print()
    
    # Check and install dependencies
    if not check_and_install_huggingface_hub():
        print("❌ Failed to install required dependencies")
        sys.exit(1)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Setup retroBERT by downloading models and dataset")
    parser.add_argument("--models-only", action="store_true", help="Download only models")
    parser.add_argument("--dataset-only", action="store_true", help="Download only dataset")
    parser.add_argument("--models-dir", type=str, default="ckpts", help="Directory for models")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Directory for dataset")
    
    args = parser.parse_args()
    
    success = True
    
    if not args.dataset_only:
        success &= download_models(args.models_dir)
    
    if not args.models_only:
        success &= download_dataset(args.dataset_dir)
    
    if success:
        if verify_setup():
            print("\n🎉 Setup completed successfully!")
            print("\nYou can now run:")
            print("  python run_retrobert.py demo_inference  # Run inference with pre-trained models")
            print("  python run_retrobert.py main            # Run full training")
            print("  python run_retrobert.py demo            # Run demo training")
        else:
            print("\n⚠️  Setup completed with some issues. Please check the verification output above.")
    else:
        print("\n❌ Setup failed. Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
