# Input data format

Two CSV formats appear in this repository. The model trains on the second; the first
is the input the second is derived from, and is included so the derivation can be
inspected and re-run.

| | `motion/` | `pose/` |
|---|---|---|
| Keypoints | 9 | 7 |
| Columns | 27 | 21 |
| Header row | yes | yes |
| Frame of reference | arena | egocentric (animal-centred) |
| Produced by | 3D tracking | `data_prep/extract_pose.py` |
| Used for | input to extraction | input to the model |

Every column is named `<keypoint>_<axis>` with axis in `x`, `y`, `z`. One row is one
frame. **Columns are looked up by name, never by position**, so your own files need the
right column *names* — the order is up to you.

## `motion/` — arena frame, 27 columns

Distributed for the male cohort at `dataset/male/motion/{preS,preR}/`.

| Columns | Keypoint | Notes |
|---|---|---|
| 0–2 | `nose` | |
| 3–5 | `head` | |
| 6–8 | `body_center` | with `tail_base`, defines the spine axis |
| 9–11 | `left_forepaw` | |
| 12–14 | `right_forepaw` | |
| 15–17 | `tail_base` | the anus; anchor for extraction |
| 18–20 | `left_hindpaw` | |
| 21–23 | `right_hindpaw` | |
| 24–26 | `tail_end` | dropped by extraction |

Coordinates are absolute positions in the arena, so they carry both where the animal
is and which way it faces. Extraction removes both.

## `pose/` — egocentric, 21 columns

What the model reads, at `dataset/<cohort>/pose/{preS,preR}/`.

| Columns | Keypoint |
|---|---|
| 0–2 | `nose` |
| 3–5 | `head` |
| 6–8 | `body_center` |
| 9–11 | `right_hindpaw` |
| 12–14 | `left_hindpaw` |
| 15–17 | `right_forepaw` |
| 18–20 | `left_forepaw` |

`tail_base` is gone because it is the origin — identically `(0, 0, 0)` in every frame,
so it carries no information. `tail_end` is dropped as the least reliably tracked point.

## The transform

For each frame, `extract_pose.py` applies:

1. **Anchor** — subtract `tail_base` from every keypoint, putting it at the origin.
2. **Heading** — `theta = atan2(body_center_y - tail_base_y, body_center_x - tail_base_x)`.
3. **Rotate** — rotate every keypoint's `(x, y)` by `-theta`, so the spine lies along `+X`
   and the animal always "faces" `+X`.
4. **Height** — `z` is shifted by `tail_base_z` but never rotated; height off the floor is
   posture, not orientation.
5. **Drop** — remove `tail_base` and `tail_end`.

The result is invariant to position and heading: two animals in identical postures at
opposite corners of the arena facing opposite directions produce identical rows. Only
posture, and how it changes over time, survives.

Re-run it, or check it against what is distributed:

```bash
python data_prep/extract_pose.py --cohort male            # motion/ -> pose/
python data_prep/extract_pose.py --cohort male --verify   # compare, write nothing
```

## Left and right

In the egocentric frame the animal faces `+X`, viewed from above (dorsal):

- **positive Y = the animal's anatomical LEFT**
- **negative Y = the animal's anatomical RIGHT**

So `left_*` keypoints have positive mean `y` and `right_*` negative. If your tracking
software labels sides by image coordinates rather than anatomy, check this before
trusting the names — a dorsal and a ventral view give opposite answers.

## Units and scale

Coordinates are in the units of the tracking output. In the distributed data the median
`tail_base`→`body_center` distance is about 3.2 and `tail_base`→`nose` about 6.0, which
is consistent with centimetres for an adult mouse.

**Absolute units do not matter.** Before the model sees a sequence, every coordinate is
divided by the animal's own spine length (`--spine_scale per_animal_median`, the default),
so the input is expressed in spine lengths. Any consistent unit works, and body-size
differences between animals are normalised away. The alternatives are `frame_wise`
(divide each frame by its own spine length) and `none`.

## Frame rate and sequence length

The distributed recordings are **11,800 frames at 20 fps** — 9.8 minutes per animal.

`--max_seq_length` is the total number of tokens the encoder sees: one `[CLS]` token
plus `max_seq_length - 1` pose frames. The default of 512 is therefore a **511-frame
window = 25.6 s** at 20 fps.

If your data has a different frame rate, match the window's *duration* rather than its
frame count:

| Frame rate | Frames for ~25.6 s | `--max_seq_length` |
|---|---|---|
| 20 fps | 511 | 512 (default) |
| 30 fps | 767 | 768 |
| 50 fps | 1279 | 1280 |
| 60 fps | 1535 | 1536 |

### Longer windows do better, up to what the GPU allows

Window length was swept at 20 fps. Pooled file-level ROC-AUC:

| Window | `--max_seq_length` | Pooled ROC-AUC |
|---|---|---|
| 32 frames | 32 | 0.645 |
| 64 frames | 64 | 0.667 |
| 128 frames | 128 | 0.670 |
| 256 frames | 256 | 0.678 |
| **512 frames** | **512** | **0.705** |

Longer is better: more of the behavioural sequence is visible at once, and the model
has more context to work with.

**But attention memory grows quadratically with window length**, so this is bounded by
VRAM, not by the data. Measured peak allocation for one training step at the default
batch size of 64, BERT-base, fp32:

| `--max_seq_length` | Peak VRAM (batch 64) |
|---|---|
| 128 | 5.3 GB |
| 256 | 10.1 GB |
| 512 | 19.7 GB |

The reported runs used 512 on a 32 GB card. On a smaller GPU, reduce `--batch_size`
before shortening the window — peak memory falls roughly linearly with batch size,
whereas shortening the window costs accuracy. A high frame rate may also be easier to
handle by downsampling towards ~20 fps than by widening the window.

## Filenames carry the cohort

Animal ids are `pre<cohort><animal>.csv` — **the 4th character of the stem is the cohort
number**, and that is what the leave-one-cohort-out split groups on:

```
pre108.csv  ->  cohort 1        pre315.csv  ->  cohort 3
```

This is load-bearing. A file named `mouse_01.csv` will fail, and a 4-digit id like
`pre1101.csv` would be read as cohort 1, not 11. Renumber your animals so the cohort is
that single character; the female cohort in this repository was renumbered for exactly
this reason.

**At least two cohorts are required**, since the number of folds *is* the number of
distinct cohorts found. There is nothing to configure: three cohorts gives three folds.

## The label table

`dataset/<cohort>/SIratio.xlsx`, one row per animal, shared by every representation of
that cohort:

| Column | Meaning |
|---|---|
| `name` | file stem, e.g. `pre108` — must match the CSV filename |
| `group` | `susceptible`, `resilient`, or `control` |
| `SI_ratio` | social interaction ratio, recorded for reference; not read by the model |

The model reads only `name` and `group`, and maps `susceptible` → 0, `resilient` → 1.
Rows for animals with no CSV present are ignored, so control rows can stay in the table
even when control recordings are not distributed.

## Bringing your own data

1. Produce 3D keypoint CSVs with a header row naming columns `<keypoint>_<x|y|z>`.
2. Make sure `tail_base` and `body_center` exist — extraction needs them for the anchor
   and the heading. Any other keypoints you keep simply become extra channels.
3. Name files `pre<cohort><animal>.csv` and place them in `preS/` (susceptible) and
   `preR/` (resilient) under a `motion/` directory.
4. Write the label table with `name` and `group`.
5. Run `python data_prep/extract_pose.py --input <your motion dir> --output <your pose dir>`.
6. Set `--max_seq_length` for your frame rate (table above) and train.

The model's input width is read from the CSVs, not declared, so a different number of
keypoints needs no code change — 7 keypoints gives 21 channels, 10 would give 30.
