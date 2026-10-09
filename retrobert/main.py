"""Entry point: leave-one-cohort-out training and evaluation."""

import os
import sys
import json
import argparse

from torch.utils.data import DataLoader

from .config import ARGS_STR, add_default_args, resolve_args
from .data import retroBERTdataset
from .inference import run_test
from .log import (print_fold_header, print_hyperparameters, print_saved,
                  print_scheduler_steps, print_summary, report_distribution,
                  start_tee, stop_tee)
from .loss import F1_Loss
from .metric import initialize_kfold_results, seed_summary
from .model import retroBERT
from .train import train
from .utils.data_utils import (prepare_datasets_with_scaling, setup_fold_directories,
                               setup_kfold_experiment)
from .utils.model_utils import set_optim, set_seed


def main():
    parser = add_default_args(argparse.ArgumentParser())
    args = parser.parse_args(ARGS_STR.split() + sys.argv[1:])
    set_seed(args)
    exp_base = args.exp_name
    resolve_args(args)

    label_file, save_dataset_dir, cohort_ids = setup_kfold_experiment(args)
    num_folds = len(cohort_ids)

    results = initialize_kfold_results()
    fold_total_steps, fold_warmup_steps = [], []

    fold_results_dir = os.path.join(args.output_dir, args.exp_name, 'fold_results')
    os.makedirs(fold_results_dir, exist_ok=True)

    for fold in range(1, num_folds + 1):
        args.save_model_path = os.path.join(args.output_dir, args.exp_name, f"fold{fold}")
        os.makedirs(args.save_model_path, exist_ok=True)
        fold_log_path = os.path.join(fold_results_dir, f'fold{fold}.txt')
        fold_log, previous_stdout = start_tee(fold_log_path)

        test_cohort = cohort_ids[fold - 1]
        print_fold_header(fold, num_folds, test_cohort, args.save_model_path)

        fold_dirs, label_maps = setup_fold_directories(fold, save_dataset_dir)

        train_dataset = retroBERTdataset(data_dir=fold_dirs['train'], label_file=label_file,
                                         is_train=True, is_test=False,
                                         shuffle_targets=(args.shuffle == 'labels'), args=args)
        valid_dataset = retroBERTdataset(data_dir=fold_dirs['valid'], label_file=label_file,
                                         is_train=False, is_test=False,
                                         shuffle_targets=(args.shuffle == 'labels'), args=args)
        test_dataset = retroBERTdataset(data_dir=fold_dirs['test'], label_file=label_file,
                                        is_train=False, is_test=True, args=args)
        train_dataset, valid_dataset, train_scaler = prepare_datasets_with_scaling(
            train_dataset, valid_dataset, args)

        train_proportions = report_distribution("Train", train_dataset)
        report_distribution("Validation", valid_dataset)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  collate_fn=valid_dataset.collate_fn, shuffle=False)

        args.input_dim = train_dataset.input_dim
        model = retroBERT(args.input_dim, args.max_seq_length).to(args.device)
        optimizer, scheduler, total_steps, warmup_steps = set_optim(model, args, train_loader)
        fold_total_steps.append(total_steps)
        fold_warmup_steps.append(warmup_steps)
        print_scheduler_steps(fold, total_steps, warmup_steps, args.warmup_ratio)

        train(model=model, train_loader=train_loader, optimizer=optimizer,
              criterion=F1_Loss(class_weights=train_proportions),
              valid_loader=valid_loader, scheduler=scheduler, args=args)

        report_distribution("Test", test_dataset)
        run_test(model, fold, fold_dirs, label_maps, test_dataset, train_scaler,
                 args, results)

        stop_tee(fold_log, previous_stdout)
        print_saved(f"fold {fold} results", fold_log_path)

    summary_path = os.path.join(fold_results_dir, 'summary.txt')
    summary_log, previous_stdout = start_tee(summary_path)

    print_hyperparameters(args, F1_Loss.__name__, fold_warmup_steps, fold_total_steps)
    print_summary(results, num_folds)

    stop_tee(summary_log, previous_stdout)
    print_saved("run summary", summary_path)

    metrics_path = os.path.join(fold_results_dir, 'summary_metrics.json')
    with open(metrics_path, 'w') as fh:
        json.dump({
            'exp_name': args.exp_name,
            'exp_base': exp_base,
            'seed': args.seed,
            'cohort': args.cohort,
            'num_folds': num_folds,
            'max_seq_length': args.max_seq_length,
            'input_dim': args.input_dim,
            'shuffle': args.shuffle if args.shuffle == 'none'
                       else f"{args.shuffle} (seed={args.shuffle_seed})",
            'per_fold': {k: results[k] for k in ('file_acc', 'file_f1', 'roc_auc')},
            'seed_mean': seed_summary(results),
        }, fh, indent=2)
    print_saved("seed metrics", metrics_path)


if __name__ == "__main__":
    main()
