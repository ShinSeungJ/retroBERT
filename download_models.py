#!/usr/bin/env python3
"""
Download checkpoints from Hugging Face Hub
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("❌ huggingface_hub not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download, snapshot_download

def download_models(target_dir="ckpts"):
    """
    Download checkpoints from Hugging Face Hub
    
    Args:
        target_dir (str): Directory to save the models (default: "ckpts")
    """
    print("🤗 Downloading checkpoints from Hugging Face...")
    print("=" * 60)
    
    repo_id = "ShinSeungJ/retroBERT"
    
    try:
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Download the entire ckpts directory
        print(f"📥 Downloading models to: {target_dir}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=target_dir,
            allow_patterns="ckpts/**",
            local_dir_use_symlinks=False
        )
        
        # Move files from ckpts/ckpts to ckpts (flatten structure)
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
        print(f"   Location: {os.path.abspath(target_dir)}")
        
        # Verify download
        expected_folds = ["fold1", "fold2", "fold3", "fold4"]
        for fold in expected_folds:
            fold_path = os.path.join(target_dir, fold)
            if os.path.exists(fold_path):
                model_file = os.path.join(fold_path, "checkpoint_best_f1.pth.tar")
                if os.path.exists(model_file):
                    print(f"   ✓ {fold}/checkpoint_best_f1.pth.tar")
                else:
                    print(f"   ⚠️  {fold}/checkpoint_best_f1.pth.tar not found")
            else:
                print(f"   ❌ {fold} directory not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading models: {str(e)}")
        print("   Please check your internet connection and try again.")
        return False

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download retroBERT checkpoints")
    parser.add_argument("--output-dir", type=str, default="ckpts",
                       help="Directory to save models (default: ckpts)")
    
    args = parser.parse_args()
    
    success = download_models(args.output_dir)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
