#!/usr/bin/env python3
"""
CLI script for retroBERT - Stress Susceptibility Prediction using BERT-based Deep Learning
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def check_and_download_requirements():
    """Check if models and dataset exist, offer to download if missing"""
    models_exist = os.path.exists("ckpts") and any(
        os.path.exists(os.path.join("ckpts", f"fold{i}", "checkpoint_best_f1.pth.tar")) 
        for i in range(1, 5)
    )
    
    dataset_exists = os.path.exists("dataset/fold") and any(
        os.path.exists(os.path.join("dataset/fold", f"train_fold_{i}"))
        for i in range(1, 5)
    )
    
    if not models_exist or not dataset_exists:
        print("⚠️  Missing required files:")
        if not models_exist:
            print("   - Pre-trained models (ckpts directory)")
        if not dataset_exists:
            print("   - Dataset (dataset/fold directory)")
        print()
        
        response = input("Would you like to download them automatically from Hugging Face? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("\n📥 Downloading required files...")
            try:
                result = subprocess.run([sys.executable, "setup_retrobert.py"], check=True)
                print("✅ Download completed!")
                return True
            except subprocess.CalledProcessError:
                print("❌ Download failed. Please run 'python setup_retrobert.py' manually.")
                return False
        else:
            print("❌ Cannot proceed without required files.")
            print("   Run 'python setup_retrobert.py' to download them.")
            return False
    
    return True

def validate_path(path, path_type="directory"):
    """Validate if path exists"""
    if not os.path.exists(path):
        print(f"Error: {path_type.capitalize()} '{path}' does not exist.")
        return False
    return True

def main_command(args):
    """Run full training with k-fold cross-validation (calls main.py)"""
    print("🚀 Starting retroBERT Full Training (K-Fold Cross-Validation)...")
    print("=" * 60)
    print("This will run the complete training pipeline from main.py")
    
    # Check if dataset exists, offer to download if missing
    if not os.path.exists("dataset/fold"):
        print("⚠️  Dataset not found.")
        response = input("Would you like to download the dataset from Hugging Face? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            try:
                result = subprocess.run([sys.executable, "setup_retrobert.py", "--dataset-only"], check=True)
                print("✅ Dataset download completed!")
            except subprocess.CalledProcessError:
                print("❌ Dataset download failed.")
                return
        else:
            print("❌ Cannot proceed without dataset.")
            return
    
    # Validate dataset directory
    if not validate_path(args.data_dir):
        print("Please ensure your dataset directory contains:")
        print("  - SIratio.xlsx (labels file)")
        print("  - fold/ directory with k-fold split data")
        return
    
    # Set up environment variables or arguments
    main_args = [
        sys.executable, "-m", "retrobert.main"
    ]
    
    # Add custom arguments if provided
    if args.epochs:
        main_args.extend(["--train_epochs", str(args.epochs)])
    if args.batch_size:
        main_args.extend(["--train_batch_size", str(args.batch_size)])
    if args.learning_rate:
        main_args.extend(["--learning_rate", str(args.learning_rate)])
    if args.output_dir:
        main_args.extend(["--output_dir", args.output_dir])
    if args.exp_name:
        main_args.extend(["--exp_name", args.exp_name])
    
    try:
        # Change to data directory if specified
        original_cwd = os.getcwd()
        if args.data_dir != ".":
            os.chdir(args.data_dir)
        
        # Run main.py
        result = subprocess.run(main_args, check=True)
        print("\n✅ Main training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        return
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        return
    finally:
        # Restore original directory
        os.chdir(original_cwd)

def demo_inference_command(args):
    """Run inference with pre-trained models (calls demo_inference.py)"""
    print("🔍 Starting retroBERT Demo Inference...")
    print("=" * 60)
    print("This will run the demo_inference.py script with pre-trained models")
    
    # Check and download requirements if needed
    if not check_and_download_requirements():
        return
    
    # Validate paths
    if not validate_path(args.data_dir):
        return
    
    # Run inference
    inference_args = [sys.executable, "-m", "retrobert.demo_inference"]
    
    try:
        # Change to data directory if specified
        original_cwd = os.getcwd()
        if args.data_dir != ".":
            os.chdir(args.data_dir)
        
        result = subprocess.run(inference_args, check=True)
        print("\n✅ Demo inference completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Inference failed with error code {e.returncode}")
        return
    except Exception as e:
        print(f"\n❌ Error during inference: {str(e)}")
        return
    finally:
        os.chdir(original_cwd)

def demo_command(args):
    """Run training with demo dataset only"""
    print("🎯 Starting retroBERT Demo Training...")
    print("=" * 60)
    print("This will train a model using only the demo dataset.")
    print("Note: This is for demonstration purposes only - not for production use.")
    print()
    
    # Check if demo dataset exists
    demo_data_dir = "dataset/demo"
    if not validate_path(demo_data_dir):
        print("Demo dataset not found. Creating minimal demo dataset...")
        create_demo_dataset()
    
    # Create demo training script
    demo_script_content = '''#!/usr/bin/env python3
"""
Demo training script for retroBERT with small dataset
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add retrobert to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrobert.data import retroBERTdataset
from retrobert.model import retroBERT
from retrobert.train import train
from retrobert.loss import F1_Loss
from retrobert.config import add_default_args
from retrobert.utils.model_utils import set_seed, set_optim
from retrobert.utils.data_utils import prepare_datasets_with_scaling
from retrobert.log import check_distribution

def main():
    """Demo training function"""
    print("🎯 retroBERT Demo Training")
    print("=" * 40)
    
    # Create demo arguments
    parser = argparse.ArgumentParser(description='retroBERT Demo Training')
    parser = add_default_args(parser)
    
    demo_args = [
        '--exp_name=demo',
        '--train_epochs=3',  # Very few epochs for demo
        '--train_batch_size=4',
        '--valid_batch_size=4',
        '--max_seq_length=32',  # Shorter sequences
        '--learning_rate=1e-5',
        '--report_every_step=1',
        '--eval_every_step=2',
        '--output_dir=outputs/demo'
    ]
    
    args = parser.parse_args(demo_args)
    set_seed(args)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {args.device}")
    
    # Demo dataset paths
    demo_unified_dir = "dataset/demo/unified"
    demo_label_file = "dataset/demo/SIratio_demo.csv"
    
    # Create unified demo directory
    os.makedirs(demo_unified_dir, exist_ok=True)
    import shutil
    for subdir in ["preS", "preR"]:
        src_dir = os.path.join("dataset/demo", subdir)
        if os.path.exists(src_dir):
            for file in os.listdir(src_dir):
                if file.endswith('.csv'):
                    shutil.copy2(
                        os.path.join(src_dir, file),
                        os.path.join(demo_unified_dir, file)
                    )
    
    try:
        # Create datasets
        print("📊 Loading demo dataset...")
        train_dataset = retroBERTdataset(
            data_dir=demo_unified_dir,
            label_file=demo_label_file,
            is_train=True,
            is_test=False,
            args=args
        )
        
        # Use same dataset for validation in demo
        valid_dataset = retroBERTdataset(
            data_dir=demo_unified_dir,
            label_file=demo_label_file,
            is_train=False,
            is_test=False,
            args=args
        )
        
        print(f"✅ Dataset loaded: {len(train_dataset)} samples")
        
        # Check distribution
        print("\\n📈 Dataset Distribution:")
        check_distribution(train_dataset)
        
        # Apply scaling
        train_dataset, valid_dataset, train_scaler = prepare_datasets_with_scaling(train_dataset, valid_dataset)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, 
                                 collate_fn=train_dataset.collate_fn, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size,
                                 collate_fn=valid_dataset.collate_fn, shuffle=False)
        
        # Create model
        print("\\n🏗️ Creating model...")
        input_dim = 24
        bert_dim = 768
        model = retroBERT(input_dim, bert_dim, args.max_seq_length)
        model.to(args.device)
        
        # Setup training
        step, best_f1 = 0, -1
        optimizer, scheduler = set_optim(model, args, train_loader)
        criterion = F1_Loss()
        
        print("\\n🔄 Starting demo training...")
        train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            step=step,
            valid_loader=valid_loader,
            best_f1=best_f1,
            scheduler=scheduler,
            args=args
        )
        
        print("\\n🎉 Demo training completed successfully!")
        print(f"   Model saved to: {args.output_dir}")
        print("   This was a demonstration with limited data and epochs.")
        print("   For full training, use: python run_retrobert.py train --data-dir <your_data>")
        
    except Exception as e:
        print(f"\\n❌ Demo training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    # Write demo script
    with open("demo_train.py", "w") as f:
        f.write(demo_script_content)
    
    # Run demo
    try:
        result = subprocess.run([sys.executable, "demo_train.py"], check=True)
        print("\n✅ Demo completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Demo failed with error code {e.returncode}")
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
    finally:
        # Clean up demo script
        if os.path.exists("demo_train.py"):
            os.remove("demo_train.py")

def create_demo_dataset():
    """Create minimal demo dataset if it doesn't exist"""
    print("Creating minimal demo dataset...")
    
    demo_dir = "dataset/demo"
    os.makedirs(f"{demo_dir}/preS", exist_ok=True)
    os.makedirs(f"{demo_dir}/preR", exist_ok=True)
    
    # Create minimal CSV files with random data
    import numpy as np
    import pandas as pd
    
    # Create demo files
    for i in range(1, 4):
        # Susceptible samples
        data = np.random.randn(100, 24)  # 100 time points, 24 features
        df = pd.DataFrame(data)
        df.to_csv(f"{demo_dir}/preS/demo_s_{i}.csv", index=False, header=False)
        
        # Resilient samples
        data = np.random.randn(100, 24)
        df = pd.DataFrame(data)
        df.to_csv(f"{demo_dir}/preR/demo_r_{i}.csv", index=False, header=False)
    
    # Create labels file
    labels_content = """name,group
demo_s_1,susceptible
demo_s_2,susceptible
demo_s_3,susceptible
demo_r_1,resilient
demo_r_2,resilient
demo_r_3,resilient"""
    
    with open(f"{demo_dir}/SIratio_demo.csv", "w") as f:
        f.write(labels_content)
    
    print("✅ Demo dataset created successfully!")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='retroBERT CLI - Stress Susceptibility Prediction using BERT-based Deep Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full k-fold cross-validation training (calls main.py)
  python run_retrobert.py main --data-dir ./dataset

  # Train with custom parameters
  python run_retrobert.py main --data-dir ./dataset --epochs 30 --batch-size 32 --lr 1e-6

  # Run inference with pre-trained models (calls demo_inference.py)
  python run_retrobert.py demo_inference --data-dir ./dataset --model-dir ./outputs/trained

  # Run demo training (training with demo dataset only)
  python run_retrobert.py demo

For more information, see README.md
        """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Main command
    main_parser = subparsers.add_parser('main', help='Run full training with k-fold cross-validation (calls main.py)')
    main_parser.add_argument('--data-dir', type=str, default='./dataset', 
                             help='Path to dataset directory (default: ./dataset)')
    main_parser.add_argument('--epochs', type=int, help='Number of training epochs (default: from config)')
    main_parser.add_argument('--batch-size', type=int, help='Training batch size (default: from config)')
    main_parser.add_argument('--lr', '--learning-rate', type=float, dest='learning_rate',
                             help='Learning rate (default: from config)')
    main_parser.add_argument('--output-dir', type=str, help='Output directory for models (default: outputs)')
    main_parser.add_argument('--exp-name', type=str, help='Experiment name (default: retrobert)')
    
    # Demo Inference command
    demo_inference_parser = subparsers.add_parser('demo_inference', help='Run inference with pre-trained models (calls demo_inference.py)')
    demo_inference_parser.add_argument('--data-dir', type=str, default='./dataset',
                                 help='Path to dataset directory (default: ./dataset)')
    demo_inference_parser.add_argument('--model-dir', type=str, default='./outputs/trained',
                                 help='Path to pre-trained models directory (default: ./outputs/trained)')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run training with demo dataset only')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute the appropriate command
    if args.command == 'main':
        main_command(args)
    elif args.command == 'demo_inference':
        demo_inference_command(args)
    elif args.command == 'demo':
        demo_command(args)

if __name__ == "__main__":
    main()
