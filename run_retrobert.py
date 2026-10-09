#!/usr/bin/env python3
"""
CLI for retroBERT - susceptibility prediction from pre-stress pose dynamics.

Examples
--------
    # leave-one-cohort-out training, all five reported seeds (male cohort)
    python run_retrobert.py train

    # the female cohort
    python run_retrobert.py train --cohort female

    # a negative control
    python run_retrobert.py train --shuffle sequences

    # a single seed, into a named experiment directory
    python run_retrobert.py train --seeds 42 --exp-name my_run
"""

import argparse
import os
import random
import subprocess
import sys

# With no --seeds, this many seeds are drawn from [SEED_MIN, SEED_MAX].
N_RANDOM_SEEDS = 5
SEED_MIN, SEED_MAX = 0, 100

def required_data(cohort):
    base = os.path.join("dataset", cohort)
    return [os.path.join(base, "pose", "preS"),
            os.path.join(base, "pose", "preR"),
            os.path.join(base, "SIratio.xlsx")]


def validate_dataset(cohort):
    """Check that the selected pose dataset is present before starting a run."""
    missing = [p for p in required_data(cohort) if not os.path.exists(p)]
    if missing:
        print("Error: the pose dataset is incomplete. Missing:")
        for p in missing:
            print(f"   - {p}")
        print("\nExpected layout:")
        print(f"   dataset/{cohort}/pose/preS/*.csv   susceptible animals, pre-stress")
        print(f"   dataset/{cohort}/pose/preR/*.csv   resilient animals, pre-stress")
        print(f"   dataset/{cohort}/SIratio.xlsx      name, group, SI_ratio")
        return False
    return True


def train_command(args):
    """Run leave-one-cohort-out training, once per seed."""
    if not validate_dataset(args.cohort or 'male'):
        return 1

    if args.seeds:
        seeds = args.seeds
        print(f"retroBERT: leave-one-cohort-out training over {len(seeds)} seed(s): "
              f"{', '.join(map(str, seeds))}")
    else:
        seeds = random.sample(range(SEED_MIN, SEED_MAX + 1), N_RANDOM_SEEDS)
        print(f"retroBERT: leave-one-cohort-out training over "
              f"{N_RANDOM_SEEDS} seeds drawn at random from "
              f"[{SEED_MIN}, {SEED_MAX}]: {', '.join(map(str, seeds))}")

    for seed in seeds:
        cmd = [sys.executable, "-m", "retrobert.main", f"--seed={seed}"]
        if args.exp_name:
            cmd.append(f"--exp_name={args.exp_name}")
        if args.output_dir:
            cmd.append(f"--output_dir={args.output_dir}")
        if args.epochs is not None:
            cmd.append(f"--train_epochs={args.epochs}")
        if args.batch_size is not None:
            cmd.append(f"--batch_size={args.batch_size}")
        if args.learning_rate is not None:
            cmd.append(f"--learning_rate={args.learning_rate}")
        if args.max_seq_length is not None:
            cmd.append(f"--max_seq_length={args.max_seq_length}")
        if args.early_stop_patience is not None:
            cmd.append(f"--early_stop_patience={args.early_stop_patience}")
        if args.cohort:
            cmd.append(f"--cohort={args.cohort}")
        if args.shuffle:
            cmd.append(f"--shuffle={args.shuffle}")
        if args.shuffle_seed is not None:
            cmd.append(f"--shuffle_seed={args.shuffle_seed}")
        cmd.extend(args.passthrough)

        print("=" * 60)
        print(f"seed={seed}")
        print(" ".join(cmd))
        print("=" * 60)

        if args.dry_run:
            continue
        # flush first, or the child's output overtakes this header when stdout is piped
        sys.stdout.flush()
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Run failed for seed={seed} (exit {result.returncode}); stopping.")
            return result.returncode

    if args.dry_run:
        return 0
    return aggregate_seeds(seeds, args)


def aggregate_seeds(seeds, args):
    """Combine the seeds just trained into one mean +/- SD table."""
    from retrobert.aggregate import aggregate, find_metrics
    from retrobert.config import default_exp_name

    exp = args.exp_name or default_exp_name()
    output_dir = args.output_dir or 'outputs'
    print("=" * 60)
    print(f"Aggregating {len(seeds)} seed(s) of {exp}")
    print("=" * 60)
    shuffle_seed = 42 if args.shuffle_seed is None else args.shuffle_seed
    paths = find_metrics(exp, args.cohort or 'male', seeds, output_dir,
                         args.shuffle or 'none', shuffle_seed)
    return 0 if aggregate(paths) else 1


def main():
    parser = argparse.ArgumentParser(
        description="retroBERT - susceptibility prediction from pre-stress pose dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser(
        "train", help="Leave-one-cohort-out training over one or more seeds")
    train_parser.add_argument("--seeds", type=int, nargs="+",
                              help=f"Seeds to run (default: {N_RANDOM_SEEDS} drawn at "
                                   f"random from [{SEED_MIN}, {SEED_MAX}])")
    train_parser.add_argument("--exp-name", type=str,
                              help="Experiment name; the seed is appended to it")
    train_parser.add_argument("--output-dir", type=str,
                              help="Where checkpoints and fold logs are written (default: outputs)")
    train_parser.add_argument("--epochs", type=int, help="Training epochs (default: from config)")
    train_parser.add_argument("--batch-size", type=int, help="Batch size (default: from config)")
    train_parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                              help="Learning rate (default: from config)")
    train_parser.add_argument("--max-seq-length", type=int,
                              help="Total tokens the encoder sees: this many minus one "
                                   "pose frames, plus a [CLS] token (default: 512)")
    train_parser.add_argument("--cohort", type=str, choices=['male', 'female'],
                              help="Which dataset to train on (default: male)")
    train_parser.add_argument("--shuffle", type=str, choices=['none', 'labels', 'sequences'],
                              help="Negative control: 'labels' permutes susceptible/resilient, "
                                   "'sequences' destroys temporal order (default: none)")
    train_parser.add_argument("--shuffle-seed", type=int,
                              help="Seed for the active shuffle (default: 42)")
    train_parser.add_argument("--early-stop-patience", type=int,
                              help="Stop after this many epochs in which neither the "
                                   "validation loss nor F1 improved; 0 disables it "
                                   "(default: 10)")
    train_parser.add_argument("--dry-run", action="store_true",
                              help="Print the commands without running them")

    # Anything this CLI does not define is forwarded verbatim to retrobert.main,
    # so every argument in retrobert/config.py stays reachable from here.
    args, passthrough = parser.parse_known_args()
    args.passthrough = passthrough
    if not args.command:
        parser.print_help()
        return 1
    if args.command == "train":
        return train_command(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
