# retroBERT for stress susceptibility prediction

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import retroBERT
from data import retroBERTdataset
from loss import F1_Loss
from train import train
from inference import run_test
from config import add_default_args, ARGS_STR
from utils.data_utils import setup_fold_directories, prepare_datasets_with_scaling
from utils.model_utils import set_seed, set_optim
from log import check_distribution, initialize_kfold_results, collect_fold_results, print_kfold_table


def main():
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    args = parser.parse_args(ARGS_STR.split())
    set_seed(args)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.save_model_path = os.path.join(args.output_dir, args.exp_name)
        
    BASE_DATA_DIR = os.path.join("dataset")
    label_file = os.path.join(BASE_DATA_DIR, "SIratio.xlsx")
    save_dataset_dir = os.path.join(BASE_DATA_DIR, 'fold')

    results = initialize_kfold_results()
    num_folds = 4

    for fold in range(1, num_folds+1):
        # Setup fold directories and get paths
        fold_dirs, filename_label_map = setup_fold_directories(fold, save_dataset_dir)
        
        # Create datasets
        train_dataset = retroBERTdataset(data_dir=fold_dirs['train_dir'], label_file=label_file, is_train=True, is_test=False, args=args)
        valid_dataset = retroBERTdataset(data_dir=fold_dirs['valid_dir'], label_file=label_file, is_train=False, is_test=False, args=args)
        test_dataset = retroBERTdataset(data_dir=fold_dirs['test_dir'], label_file=label_file, is_train=False, is_test=True, args=args)

        train_dataset, valid_dataset, train_scaler = prepare_datasets_with_scaling(train_dataset, valid_dataset)
        
        print("Train Dataset Distribution:")
        check_distribution(train_dataset)
        print("Validation Dataset Distribution:")
        check_distribution(valid_dataset)

        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, collate_fn=valid_dataset.collate_fn, shuffle=False)

        input_dim = 24
        model = retroBERT(input_dim, args.max_seq_length)
        model.to(args.device)
        step, best_f1 = 0, -1
        optimizer, scheduler = set_optim(model, args, train_loader)
        criterion = F1_Loss()

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

        print("Test Dataset Distribution:")
        check_distribution(test_dataset)

        # Run testing and collect results
        file_acc, precision, recall, file_f1 = run_test(model, fold_dirs['test_dir'], test_dataset, filename_label_map, train_scaler, args)
        collect_fold_results(results, file_acc, precision, recall, file_f1)

    # print_final_kfold_results(results, num_folds)
    print_kfold_table(results, num_folds)

if __name__ == "__main__":
    main()