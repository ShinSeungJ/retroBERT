"""Re-score trained runs on their held-out cohorts, optionally with shuffled test windows.

Training scores the test set once, in the order its --shuffle settings dictate. This
scores the same checkpoints again without retraining, so the test order can be chosen
independently of how the model was trained:

    --test_order shuffled   frame order permuted within every held-out test window
    --test_order intact     windows in their true order (reproduces the training score)

Scoring a real run (--shuffle none) with shuffled test windows gives the
intact-train / shuffled-test cell of the sequence-shuffle design; the other cells come
from training with --shuffle sequences and --shuffle_scope fit or all.

The trained run is located by the same arguments it was trained with (--exp_name,
--cohort, --seed, --shuffle, --shuffle_seed, --shuffle_scope). Data and model settings
are read back from its checkpoints, so they match training exactly. Results go to
<run>/eval_testshuffle<test_shuffle_seed>/ or <run>/eval_intact/, next to fold_results/.

Usage
-----
    python -m retrobert.evaluate --seed=42 --test_order=shuffled --test_shuffle_seed=42
"""

import argparse
import copy
import json
import os
import sys

import torch

from .config import (ARGS_STR, add_default_args, resolve_args, shuffle_description,
                     test_order_description)
from .data import retroBERTdataset
from .inference import run_test
from .log import (print_evaluation_header, print_fold_header, print_missing_run,
                  print_saved, print_summary, tee_to_file)
from .metric import initialize_kfold_results, seed_summary
from .model import retroBERT
from .utils.data_utils import fit_scaler, setup_fold_directories, setup_kfold_experiment
from .utils.model_utils import set_seed

# The checkpoint training scores the held-out cohort with (see inference.run_test).
CHECKPOINT = 'checkpoint_best_f1.pth.tar'

# Fold copies for re-scoring live apart from training's, so an evaluation running
# alongside a training run can never rebuild the directories that run is reading.
EVAL_FOLD_DIR = 'cohort_strat_fold_eval'


def eval_dir_name(test_order, test_shuffle_seed):
    """The results directory evaluate writes inside a run."""
    if test_order == 'shuffled':
        return f"eval_testshuffle{test_shuffle_seed}"
    return 'eval_intact'


def add_eval_args(parser):
    parser.add_argument('--test_order', type=str, default='shuffled',
                        choices=['shuffled', 'intact'],
                        help="Order of the held-out test windows: 'shuffled' permutes the "
                             "frames within every window, 'intact' keeps their true order.")
    parser.add_argument('--test_shuffle_seed', type=int, default=42,
                        help='Seed for the test-window permutation, independent of the '
                             'seed the run was trained with.')
    return parser


def main():
    parser = add_eval_args(add_default_args(argparse.ArgumentParser()))
    cli = parser.parse_args(ARGS_STR.split() + sys.argv[1:])
    set_seed(cli)
    exp_base = cli.exp_name
    resolve_args(cli)

    run_dir = os.path.join(cli.output_dir, cli.exp_name)
    first_checkpoint = os.path.join(run_dir, 'fold1', CHECKPOINT)
    if not os.path.exists(first_checkpoint):
        print_missing_run(first_checkpoint)
        return 1

    # Everything that shapes the data (window length, spine scaling, scaler, split
    # ratio, cohort) comes from the checkpoint rather than this command line.
    args = torch.load(first_checkpoint, map_location='cpu', weights_only=False)['args']
    args.device = cli.device
    shuffle_test = cli.test_order == 'shuffled'

    eval_dir = os.path.join(run_dir, eval_dir_name(cli.test_order, cli.test_shuffle_seed))
    os.makedirs(eval_dir, exist_ok=True)
    shuffle = (cli.shuffle if cli.shuffle == 'none'
               else shuffle_description(cli.shuffle, cli.shuffle_seed, cli.shuffle_scope))
    test_order = test_order_description(shuffle_test, cli.test_shuffle_seed)

    label_file, save_dataset_dir, cohort_ids = setup_kfold_experiment(
        args, os.path.join(args.data_dir, EVAL_FOLD_DIR))
    num_folds = len(cohort_ids)
    results = initialize_kfold_results()
    input_dim = None

    for fold in range(1, num_folds + 1):
        args.save_model_path = os.path.join(run_dir, f"fold{fold}")
        fold_log_path = os.path.join(eval_dir, f'fold{fold}.txt')
        with tee_to_file(fold_log_path):
            print_fold_header(fold, num_folds, cohort_ids[fold - 1], args.save_model_path)
            fold_dirs, label_maps = setup_fold_directories(fold, save_dataset_dir)

            # The scaler is refitted on the fold's training animals exactly as in
            # training. Neither shuffle changes it: a label shuffle leaves the frames
            # alone, and a sequence shuffle reorders frames without changing them.
            train_dataset = retroBERTdataset(data_dir=fold_dirs['train'], label_file=label_file,
                                             is_train=True, is_test=False, args=args)
            scaler = (fit_scaler(train_dataset.source_seq, train_dataset.source_mask)
                      if args.use_standard_scaler else None)
            input_dim = train_dataset.input_dim
            del train_dataset

            # The test permutation has its own seed, so it is independent of any
            # permutation the run was trained with.
            test_args = copy.copy(args)
            test_args.shuffle_seed = cli.test_shuffle_seed
            test_dataset = retroBERTdataset(data_dir=fold_dirs['test'], label_file=label_file,
                                            is_train=False, is_test=True,
                                            shuffle_sequences=shuffle_test, args=test_args)

            model = retroBERT(input_dim, args.max_seq_length).to(args.device)
            run_test(model, fold, fold_dirs, label_maps, test_dataset, scaler,
                     test_args, results)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print_saved(f"fold {fold} results", fold_log_path)

    summary_path = os.path.join(eval_dir, 'summary.txt')
    with tee_to_file(summary_path):
        print_evaluation_header(run_dir, CHECKPOINT, shuffle, test_order, eval_dir)
        print_summary(results, num_folds)
    print_saved("evaluation summary", summary_path)

    metrics_path = os.path.join(eval_dir, 'summary_metrics.json')
    with open(metrics_path, 'w') as fh:
        json.dump({
            'exp_name': cli.exp_name,
            'exp_base': exp_base,
            'seed': cli.seed,
            'cohort': args.cohort,
            'num_folds': num_folds,
            'max_seq_length': args.max_seq_length,
            'input_dim': input_dim,
            'shuffle': shuffle,
            'test_order': test_order,
            'checkpoint': CHECKPOINT,
            'per_fold': {k: results[k] for k in ('file_acc', 'file_f1', 'roc_auc')},
            'seed_mean': seed_summary(results),
        }, fh, indent=2)
    print_saved("evaluation metrics", metrics_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
