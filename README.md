# retroBERT: Susceptibility Prediction from Pre-Stress Pose Dynamics

A BERT encoder that predicts stress susceptibility from 3D pose trajectories recorded
**before** any stress exposure, evaluated leave-one-cohort-out.

## Overview

retroBERT is a deep learning framework that leverages BERT (Bidirectional Encoder Representations from Transformers) architecture to predict stress susceptibility from behavioral time series data. The model processes sequential behavioral data and classifies subjects as either resilient or susceptible to stress.

Each animal contributes a 10-minute open-field recording, tracked as 3D keypoints and
transformed into an egocentric pose representation (tail-base anchored at the origin,
spine aligned to +X). The model consumes raw keypoint trajectories — no hand-built
behavioural features — and predicts whether the animal will later prove susceptible or
resilient to social defeat stress.

Evaluation is **leave-one-cohort-out (LOCO)**: all animals from one experimental cohort
are held out as the test set, so a prediction is never made by a model that has seen any
animal from that cohort. Five training seeds × five folds give five complete, independent
replications of the same held-out experiment.

## Quick Start

```bash
# install
conda create -n retrobert python=3.10 -y
conda activate retrobert
pip install -e .

# check the install end to end: one epoch, one seed (see "Smoke test")
python run_retrobert.py train --cohort female --seeds 42 --epochs 1

# leave-one-cohort-out training over 5 randomly drawn seeds, then the
# across-seed summary (male cohort by default)
python run_retrobert.py train

# the female cohort
python run_retrobert.py train --cohort female

# specific seeds instead of random ones
python run_retrobert.py train --seeds 42 36 12 48 30

# see what would run, without running it
python run_retrobert.py train --dry-run
```

## System Requirements

### Software Dependencies

| Package | Version used |
|---|---|
| Python | 3.10 |
| torch | 2.11.0 (CUDA 12.8) |
| transformers | 5.8.0 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| openpyxl | 3.1.5 |

Exact pins are in `requirements.txt`.

### Hardware

A CUDA GPU is strongly recommended. The reported runs used 512-token windows (511 pose
frames plus a `[CLS]` token) with a 12-layer BERT encoder at batch size 64; one seed (5 folds × 100 epochs) takes several
hours on a single modern GPU. The code falls back to CPU, but a full run is impractical
there.

## Smoke test

Before committing to a full run, check that the pipeline works end to end on your
machine:

```bash
python run_retrobert.py train --cohort female --seeds 42 --epochs 1
```

One seed, one epoch, on the smaller of the two cohorts — so it exercises every stage
(fold construction, the leave-one-cohort-out split, training, checkpointing, held-out
evaluation, the summary table, the per-seed metrics file and the across-seed aggregate)
without the cost of a real run.

It worked if you see, in order:

```
retroBERT: leave-one-cohort-out training over 1 seed(s): 42
  Fold 1: test=cohort 1 (18S/4R)  |  train=6S/5R  valid=2S/2R  (from cohorts [2])
  Fold 2: test=cohort 2 (8S/7R)   |  train=13S/3R valid=5S/1R  (from cohorts [1])
...
K-FOLD 2 SUMMARY (mean +/- std across folds)
...
AGGREGATE OVER 1 SEED(S) - retrobert_pose
```

and, on disk:

```
outputs/retrobert_pose_female_42/
├── fold1/ fold2/                  checkpoint_best_f1.pth.tar
└── fold_results/
    ├── fold1.txt  fold2.txt
    ├── summary.txt
    └── summary_metrics.json
```

**The numbers it prints are meaningless.** One epoch at the default learning rate of
1e-6 leaves the model essentially untrained, and the aggregate over a single seed
reports a standard deviation of 0.00 because there is nothing to vary. The test tells
you the code runs, the data is readable and the plumbing is connected — nothing more.

Swap in `--cohort male` to exercise the five-fold split as well. Run time depends
entirely on your GPU; the female cohort is faster because it has two folds and fewer
animals.

## Dataset

Two datasets ship with the package, one per sex, selected with `--cohort`:

```
dataset/
├── male/
│   ├── motion/        arena-frame 3D keypoints, 27 cols  (57 CSVs)
│   │   ├── preS/      susceptible animals, pre-stress    (19 CSVs)
│   │   └── preR/      resilient animals, pre-stress      (38 CSVs)
│   ├── pose/          egocentric, 21 cols - what the model reads
│   │   ├── preS/                                         (19 CSVs)
│   │   └── preR/                                         (38 CSVs)
│   └── SIratio.xlsx   name, group, SI_ratio              (76 rows, 5 cohorts)
└── female/
    ├── pose/
    │   ├── preS/                                         (26 CSVs)
    │   └── preR/                                         (11 CSVs)
    └── SIratio.xlsx   same three columns                 (52 rows, 2 cohorts)
```

**[`data_prep/KEYPOINTS.md`](data_prep/KEYPOINTS.md) is the full input specification** —
every column of both formats, the egocentric transform and why each step is there, the
left/right convention, units and scaling, how to choose the window length for your frame
rate, and what your own files need to look like. Read it before using this pipeline on
new data.

Both label tables use the same schema, so one code path reads either. The female
`group` column is the cohort-wise z-score call at threshold −1.5.

Animal ids follow `pre<cohort><animal>`, so the filename carries the cohort: male
animals span cohorts 1–5, female animals cohorts 1–2. The three smaller female groups
are held out together and were renumbered into a single cohort, which is why the female
experiment has two folds and the male five.

**The fold count is not configured** — it is the number of distinct cohorts found in
the filenames. Add a dataset with three cohorts and it trains three-fold with no code
change.

Each CSV is one animal: 11,800 frames at 20 fps (9.8 min) × 21 columns, i.e. 7 keypoints
× (x, y, z) in the egocentric frame:

| Keypoint | Columns |
|---|---|
| nose | 0–2 |
| head | 3–5 |
| body_center | 6–8 |
| right_hindpaw | 9–11 |
| left_hindpaw | 12–14 |
| right_forepaw | 15–17 |
| left_forepaw | 18–20 |

`tail_base` is the anchor point and is dropped from the output (it is always `0,0,0`
after the transform). The cohort is read from the filename, and is what the
leave-one-cohort-out split groups on. See
[`data_prep/KEYPOINTS.md`](data_prep/KEYPOINTS.md) for the arena-frame layout, the
transform, and the left/right convention.

### Regenerating the pose dataset

The distributed `pose/` CSVs are derived from `motion/` and can be rebuilt or checked:

```bash
python data_prep/extract_pose.py --cohort male            # motion/ -> pose/
python data_prep/extract_pose.py --cohort male --verify   # compare, write nothing
```

Columns are matched by name, so your own CSVs need the right column names rather than a
particular column order.

### Using your own data

1. Export 3D keypoints with a header row naming columns `<keypoint>_<x|y|z>`; `tail_base`
   and `body_center` must be present, since they set the anchor and the heading.
2. Name files `pre<cohort><animal>.csv` — the 4th character is the cohort, and at least
   two cohorts are needed because folds *are* cohorts.
3. Write a label table with `name` and `group` (`susceptible` / `resilient`).
4. Run `extract_pose.py`, then set `--max_seq_length` for your frame rate.

The model's input width is read from the CSVs rather than declared, so a different
number of keypoints needs no code change.
[`data_prep/KEYPOINTS.md`](data_prep/KEYPOINTS.md) has the details.

The fold directories the training script builds (`dataset/<cohort>/pose/cohort_strat_fold/`) are
derived and regenerated on every run — they are not part of the distributed dataset.

## Usage

### Training

```bash
# 5 random seeds, defaults from retrobert/config.py
python run_retrobert.py train

# override anything the module accepts
python run_retrobert.py train --seeds 42 --epochs 50 --max-seq-length 256
```

With no `--seeds`, five seeds are drawn at random from 0–100 and printed before the run
starts, so a result is never tied to one hand-picked seed. Pass `--seeds` to reproduce a
particular set.

Or call the module directly:

```bash
python -m retrobert.main --seed=42
```

Each run writes to `outputs/<exp_name>_<cohort>_<seed>/`, so male and female runs
of the same seed never overwrite each other:

```
outputs/<exp_name>_<cohort>_<seed>/
├── fold1/ ... fold5/        checkpoint_best_f1.pth.tar
└── fold_results/
    ├── fold1.txt ... fold5.txt    per-fold console log
    ├── summary.txt               hyperparameters + the cross-fold table
    └── summary_metrics.json       that seed's numbers, for the aggregator
```

### Aggregating seeds

`train` aggregates automatically once its seeds finish, reporting mean ± **sample** SD
(ddof=1) across seeds — the form the results are quoted in. The table inside a single
seed's `summary.txt` instead shows the spread across that seed's folds.

To re-aggregate later, or to combine a different set of seeds:

```bash
python -m retrobert.aggregate --exp retrobert_pose --cohort male --seeds 17 23 41 68 92
python -m retrobert.aggregate --exp retrobert_pose --cohort female --seeds 17 23 41 68 92 --shuffle labels --shuffle_seed 42
python -m retrobert.aggregate outputs/*/fold_results/summary_metrics.json
```

```bash
# stop after 5 epochs without improvement
python run_retrobert.py train --seeds 42 --early-stop-patience 5

# never stop early
python run_retrobert.py train --early-stop-patience 0
```

Because checkpoint selection already keeps the best model, early stopping changes run
time rather than which model is evaluated — unless it cuts the run short before a later
improvement would have arrived.

### Shuffle controls

Two negative controls calibrate how much of the reported signal could arise by chance:

- **Label shuffle** — animal↔label assignments are permuted while class counts are
  preserved. Any above-chance performance is spurious by construction.
- **Sequence shuffle** — the temporal order of frames/windows is destroyed while the
  pose distribution is preserved, testing whether the model uses behavioural *dynamics*
  rather than static posture statistics.

Both are flags on the same entry point, so a control runs the identical model, splits
and evaluation as the real experiment — only the thing being ablated differs:

```bash
# label shuffle: train/validation labels permuted, held-out test labels stay true
python run_retrobert.py train --shuffle labels --shuffle-seed 42

# sequence shuffle: frame order destroyed within each window
python run_retrobert.py train --shuffle sequences --shuffle-seed 42
```

| Control | What is permuted |
|---|---|
`--shuffle labels` | each fitting split's animal↔label links, after loading |
`--shuffle sequences` | frame order within every window, per-window deterministic seed |

Class counts are preserved, controls are never relabelled, and each permutation is
deterministic given `--shuffle_seed`, which is independent of `--seed`: build the null
distribution by varying the shuffle while holding the model seed fixed. Each control
writes to its own experiment directory (`..._shuffle42`, `..._seqshuffle42`), and a label
shuffle also gets its own fold directory, so a control can never overwrite the real run.

## Repository Structure

```
.
├── retrobert/                 the package
│   ├── __init__.py
│   ├── config.py              defaults, cohort registry, argument resolution
│   ├── data.py                pose CSVs -> padded, masked sequence tensors
│   ├── model.py               retroBERT encoder
│   ├── loss.py                F1_Loss
│   ├── train.py               training loop, validation, early stopping
│   ├── inference.py           per-animal prediction and held-out evaluation
│   ├── metric.py              metric computation (no printing)
│   ├── log.py                 all console output (no metric computation)
│   ├── main.py                leave-one-cohort-out entry point
│   ├── aggregate.py           combine seeds into one mean ± SD table
│   └── utils/
│       ├── __init__.py
│       ├── data_utils.py      fold construction, scaling, pose normalization
│       └── model_utils.py     seeding, checkpoints, optimizer/scheduler
├── data_prep/
│   ├── extract_pose.py        motion/ -> pose/, matched by column name
│   └── KEYPOINTS.md           input data specification
├── dataset/                   motion + pose CSVs, label tables
├── run_retrobert.py           CLI
├── requirements.txt
├── MANIFEST.in
├── LICENSE
└── setup.py
```

`metric.py` computes and never prints; `log.py` prints and never computes a metric.
That split is deliberate, so a change to the report can never change a number.

## Reported Configuration

The defaults in `retrobert/config.py` are the configuration the reported results were
produced with:

| Parameter | Value |
|---|---|
| Loss | F1_Loss |
| Epochs | 100 |
| Batch size | 64 |
| Learning rate | 1e-6 |
| Weight decay | 1e-8 |
| Warmup | 10% of total steps |
| Gradient accumulation | 1 |
| Max sequence length | 512 tokens = 511 frames (25.6 s) + 1 `[CLS]` |
| Max grad norm | 1.0 |
| Spine scale | per_animal_median |
| Standard scaler | True |
| Train/val ratio | 0.75 |
| Seeds | 5 drawn at random from 0–100 (`--seeds` to fix them) |
| Early stopping | patience 10 epochs, on validation loss and F1 |
| Shuffle control | none (`--shuffle none`) |
| Cohort | male (`--cohort male`), 5 folds |


## License

MIT — see `LICENSE`.
