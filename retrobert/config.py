"""Default hyper-parameters and the argument parser."""

import argparse
import os

import torch

# Datasets this package can train on. Each holds its label table plus one
# subdirectory per representation; the model trains on 'pose'. Fold count is not
# declared here: it is the number of distinct cohorts found in the filenames.
COHORT_DIRS = {
    'male': os.path.join('dataset', 'male'),
    'female': os.path.join('dataset', 'female'),
}
REPRESENTATION = 'pose'

# Anything passed on the command line overrides these.
ARGS_STR = f"""
--exp_name=retrobert_pose \
--train_epochs=100 \
--batch_size=64 \
--gradient_accumulation_steps=1 \
--learning_rate=1e-6 \
--warmup_percent=10 \
--weight_decay=1e-8 \
--adam_epsilon=1e-8 \
--seed=42 \
--max_seq_length=512 \
--max_grad_norm=1.0 \
--spine_scale=per_animal_median \
--use_standard_scaler=True \
--train_val_ratio=0.75 \
--early_stop_patience=10 \
"""

def add_default_args(parser):
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--warmup_percent', type=float, default=10.0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--save_every_epoch', type=bool, default=False)
    parser.add_argument('--report_every_step', type=int, default=50)
    parser.add_argument('--eval_every_step', type=int, default=50)
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Total sequence length the encoder sees: '
                             '(max_seq_length - 1) pose frames plus one [CLS] token.')
    parser.add_argument('--spine_scale', type=str, default='frame_wise',
                        choices=['frame_wise', 'per_animal_median', 'none'])
    parser.add_argument('--use_standard_scaler', type=str, default='true')
    parser.add_argument('--train_val_ratio', type=float, default=0.75)
    parser.add_argument('--cohort', type=str, default='male',
                        choices=sorted(COHORT_DIRS),
                        help='Which dataset to train on. Selects '
                             'dataset/<cohort>/pose/, alongside that cohort\'s label table.')
    parser.add_argument('--shuffle', type=str, default='none',
                        choices=['none', 'labels', 'sequences'],
                        help="Negative control. 'labels' permutes the susceptible/resilient "
                             "assignment of the fitting splits, leaving the held-out test "
                             "labels true; 'sequences' destroys temporal order within each "
                             "window. 'none' is the real experiment.")
    parser.add_argument('--shuffle_seed', type=int, default=42,
                        help='Seed for the active shuffle, independent of --seed.')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Stop training when neither the validation loss nor F1 has '
                             'improved for this many epochs. 0 disables early stopping.')
    return parser


def resolve_args(args):
    """Coerce string flags, validate warmup and derive the run's paths and device."""
    args.use_standard_scaler = str(args.use_standard_scaler).lower() in ('true', '1', 'yes')

    if args.warmup_percent < 0:
        raise ValueError("--warmup_percent must be >= 0")
    args.warmup_ratio = args.warmup_percent / 100.0 if args.warmup_percent > 1 else args.warmup_percent
    if args.warmup_ratio > 1:
        raise ValueError("--warmup_percent must be <= 100 (or <=1.0 when passed as ratio)")

    args.cohort_dir = COHORT_DIRS[args.cohort]
    args.data_dir = os.path.join(args.cohort_dir, REPRESENTATION)

    if args.max_seq_length < 2:
        raise ValueError("--max_seq_length must be >= 2 (one [CLS] token plus >=1 frame)")
    args.num_frames = args.max_seq_length - 1

    args.exp_name = run_name(args.exp_name, args.cohort, args.seed,
                             args.shuffle, args.shuffle_seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.save_model_path = os.path.join(args.output_dir, args.exp_name)
    return args


def run_name(exp, cohort, seed, shuffle='none', shuffle_seed=42):
    """The output directory name of one run. The aggregator looks runs up by this exact
    name, so a shuffle control can never be mistaken for the real run of the same seed,
    and the cohort is part of it so a male and a female run never share a directory."""
    name = f"{exp}_{cohort}_{seed}"
    if shuffle == 'labels':
        name = f"{name}_shuffle{shuffle_seed}"
    elif shuffle == 'sequences':
        name = f"{name}_seqshuffle{shuffle_seed}"
    return name


def default_exp_name():
    """The experiment name from ARGS_STR, before the seed suffix is appended."""
    tokens = ARGS_STR.split()
    for token in tokens:
        if token.startswith('--exp_name='):
            return token.split('=', 1)[1]
    return 'retrobert'
