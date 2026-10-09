"""Combine several seeds of one experiment into a single mean +/- SD table.

Each training run writes outputs/<exp>_<cohort>_<seed>/fold_results/summary_metrics.json.
This reads those files and reports the spread across seeds, which is the form the
results are quoted in.

Usage
-----
    # every seed of one experiment
    python -m retrobert.aggregate --exp retrobert_pose --cohort male --seeds 17 23 41 68 92

    # or point it straight at the json files
    python -m retrobert.aggregate outputs/*/fold_results/summary_metrics.json
"""

import argparse
import json
import os
import sys

from .log import print_aggregate


def find_metrics(exp, cohort, seeds, output_dir, shuffle='none', shuffle_seed=42):
    """Locate one summary_metrics.json per seed, by exact run name.

    No wildcard: a pattern like <exp>_42* would also match the shuffle controls of
    seed 42, and <exp>_4* would match seed 42.
    """
    from .config import run_name

    paths = []
    for seed in seeds:
        path = os.path.join(output_dir, run_name(exp, cohort, seed, shuffle, shuffle_seed),
                            'fold_results', 'summary_metrics.json')
        if not os.path.exists(path):
            print(f"  seed {seed}: no metrics found at {path} -- skipped")
            continue
        paths.append(path)
    return paths


def load(paths):
    records = []
    for p in paths:
        with open(p) as fh:
            records.append(json.load(fh))
    records.sort(key=lambda r: r['seed'])
    return records


def aggregate(paths):
    """Print the across-seed table. Returns the records it used."""
    records = load(paths)
    if not records:
        print("No seed metrics to aggregate.")
        return []
    bases = {r['exp_base'] for r in records}
    if len(bases) > 1:
        print(f"Warning: mixing experiments {sorted(bases)} in one aggregate.")
    print_aggregate(records)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='*', help='summary_metrics.json files to aggregate')
    ap.add_argument('--exp', type=str, help='Experiment name, without the seed suffix')
    ap.add_argument('--seeds', type=int, nargs='+', help='Seeds to aggregate')
    ap.add_argument('--cohort', type=str, default='male', choices=['male', 'female'])
    ap.add_argument('--output_dir', type=str, default='outputs')
    ap.add_argument('--shuffle', type=str, default='none',
                    choices=['none', 'labels', 'sequences'],
                    help='Aggregate this shuffle control instead of the real runs')
    ap.add_argument('--shuffle_seed', type=int, default=42)
    opt = ap.parse_args()

    paths = opt.paths
    if not paths:
        if not (opt.exp and opt.seeds):
            ap.error("give json paths, or --exp together with --seeds")
        paths = find_metrics(opt.exp, opt.cohort, opt.seeds, opt.output_dir,
                             opt.shuffle, opt.shuffle_seed)

    return 0 if aggregate(paths) else 1


if __name__ == "__main__":
    sys.exit(main())
