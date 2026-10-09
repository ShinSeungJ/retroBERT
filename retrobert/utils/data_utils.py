"""Fold construction, scaling and pose normalization."""

import os
import shutil
import random
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler



def get_csv_files(data_dir):
    return [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith('.csv')]

def recreate_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def create_k_fold_directories(base_dir, num_folds):
    recreate_directory(base_dir)
    for i in range(num_folds):
        os.makedirs(os.path.join(base_dir, f'train_fold_{i+1}'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'train_fold_{i+1}', '__preS'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'train_fold_{i+1}', '__preR'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'valid_fold_{i+1}'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'valid_fold_{i+1}', '__preS'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'valid_fold_{i+1}', '__preR'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'test_fold_{i+1}'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'test_fold_{i+1}', '__preS'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'test_fold_{i+1}', '__preR'), exist_ok=True)

def cohort_of(filepath):
    """Cohort id from the filename: 'pre<cohort><animal>.csv' -> int(cohort).

    Both datasets follow this convention, so the female groups that are held out
    together were renumbered into one cohort (see the rename mapping).
    """
    return int(os.path.basename(filepath)[3])

def group_files_by_cohort(file_paths):
    cohorts = {}
    for fp in file_paths:
        cohorts.setdefault(cohort_of(fp), []).append(fp)
    return cohorts

def split_cohort_test_stratified_trainval(susceptible_files, resilient_files, base_dir, args):
    sus_cohorts = group_files_by_cohort(susceptible_files)
    res_cohorts = group_files_by_cohort(resilient_files)

    cohort_ids = sorted(set(sus_cohorts.keys()) | set(res_cohorts.keys()))
    num_folds = len(cohort_ids)

    rng = np.random.RandomState(42)

    for fold_idx in range(num_folds):
        test_cohort = cohort_ids[fold_idx]
        remaining_cohorts = [c for c in cohort_ids if c != test_cohort]

        remaining_sus = []
        remaining_res = []
        for c in remaining_cohorts:
            remaining_sus.extend(sus_cohorts.get(c, []))
            remaining_res.extend(res_cohorts.get(c, []))

        remaining_sus = list(remaining_sus)
        remaining_res = list(remaining_res)
        rng.shuffle(remaining_sus)
        rng.shuffle(remaining_res)

        n_train_sus = int(len(remaining_sus) * args.train_val_ratio)
        n_train_res = int(len(remaining_res) * args.train_val_ratio)

        train_sus = remaining_sus[:n_train_sus]
        valid_sus = remaining_sus[n_train_sus:]
        train_res = remaining_res[:n_train_res]
        valid_res = remaining_res[n_train_res:]

        test_sus = sus_cohorts.get(test_cohort, [])
        test_res = res_cohorts.get(test_cohort, [])

        fold_num = fold_idx + 1
        for fp in train_sus:
            shutil.copy(fp, os.path.join(base_dir, f'train_fold_{fold_num}', '__preS', os.path.basename(fp)))
        for fp in train_res:
            shutil.copy(fp, os.path.join(base_dir, f'train_fold_{fold_num}', '__preR', os.path.basename(fp)))
        for fp in valid_sus:
            shutil.copy(fp, os.path.join(base_dir, f'valid_fold_{fold_num}', '__preS', os.path.basename(fp)))
        for fp in valid_res:
            shutil.copy(fp, os.path.join(base_dir, f'valid_fold_{fold_num}', '__preR', os.path.basename(fp)))
        for fp in test_sus:
            shutil.copy(fp, os.path.join(base_dir, f'test_fold_{fold_num}', '__preS', os.path.basename(fp)))
        for fp in test_res:
            shutil.copy(fp, os.path.join(base_dir, f'test_fold_{fold_num}', '__preR', os.path.basename(fp)))

        print(
            f"  Fold {fold_num}: test=cohort {test_cohort} "
            f"({len(test_sus)}S/{len(test_res)}R)  |  "
            f"train={len(train_sus)}S/{len(train_res)}R  "
            f"valid={len(valid_sus)}S/{len(valid_res)}R  "
            f"(from cohorts {remaining_cohorts})"
        )

    return cohort_ids

def consolidate_files(susceptible_test_dir, resilient_test_dir, unified_test_dir):
    os.makedirs(unified_test_dir, exist_ok=True)
    for file in os.listdir(susceptible_test_dir):
        if file.endswith('.csv'):
            shutil.copy(os.path.join(susceptible_test_dir, file), os.path.join(unified_test_dir, file))
    for file in os.listdir(resilient_test_dir):
        if file.endswith('.csv'):
            shutil.copy(os.path.join(resilient_test_dir, file), os.path.join(unified_test_dir, file))

def fit_scaler(source_seq, source_mask):
    all_data = torch.cat([tensor for tensor in source_seq], dim=0).numpy()
    # source_mask[:, 0] is the CLS token; columns 1: are frame-level (1=real, 0=padded)
    frame_mask = source_mask[:, 1:].reshape(-1).numpy().astype(bool)
    scaler = StandardScaler()
    scaler.fit(all_data[frame_mask])
    return scaler

def apply_scaler(source_seq, train_scaler):
    scaled_data = []
    for tensor in source_seq:
        scaled_tensor = train_scaler.transform(tensor.numpy())
        scaled_data.append(torch.tensor(scaled_tensor, dtype=torch.float32))
    return scaled_data

def normalize_spine(data_array, method):
    """Normalize coordinates by spine length (body_center distance from origin).

    Pose column layout (21 cols, 7 KPs × 3 axes, tail_base and tail_end removed):
        nose(0-2), head(3-5), body_center(6-8), right_hindpaw(9-11),
        left_hindpaw(12-14), right_forepaw(15-17), left_forepaw(18-20)
    tail_base is at the origin (0,0,0) — anchored and removed by extraction.

    Methods:
        'frame_wise'        : divide each frame by its own spine length
        'per_animal_median' : divide all frames by the animal's median spine length
        'none'              : no normalization
    """
    if method == 'none':
        return data_array

    body = data_array[:, 6:9]   # body_center (KP3 in raw, position 2 in output)
    spine_lengths = np.linalg.norm(body, axis=1)   # distance from origin = tail_base

    if method == 'frame_wise':
        spine_length = spine_lengths.reshape(-1, 1)
        data_array = data_array / spine_length
    elif method == 'per_animal_median':
        median_spine = np.median(spine_lengths)
        data_array = data_array / median_spine

    return data_array

def map_filenames_to_labels(susceptible_dir, resilient_dir):
    filename_label_map = {}
    for file in os.listdir(susceptible_dir):
        if file.endswith('.csv'):
            filename_label_map[file] = 0
    for file in os.listdir(resilient_dir):
        if file.endswith('.csv'):
            filename_label_map[file] = 1
    return filename_label_map

def setup_kfold_experiment(args):
    """Build the leave-one-cohort-out fold directories for the selected dataset.

    The number of folds is the number of distinct cohorts found in the filenames, so a
    dataset with two cohorts yields two folds without anything being declared.

    Returns (label_file, save_dataset_dir, cohort_ids).
    """
    label_file = os.path.join(args.cohort_dir, "SIratio.xlsx")
    save_dataset_dir = os.path.join(args.data_dir, 'cohort_strat_fold')

    # A label-shuffle run gets its own fold directory, so it can never race or
    # overwrite the real run's splits.
    if args.shuffle == 'labels':
        save_dataset_dir = f"{save_dataset_dir}_shuffle{args.shuffle_seed}"

    susceptible_files = get_csv_files(os.path.join(args.data_dir, 'preS'))
    resilient_files = get_csv_files(os.path.join(args.data_dir, 'preR'))

    num_cohorts = len({cohort_of(f) for f in susceptible_files + resilient_files})
    create_k_fold_directories(save_dataset_dir, num_cohorts)

    cohort_ids = split_cohort_test_stratified_trainval(
        susceptible_files, resilient_files, save_dataset_dir, args)

    return label_file, save_dataset_dir, cohort_ids


def setup_fold_directories(fold, save_dataset_dir):
    """Consolidate this fold's preS/preR splits into unified directories.

    Returns (fold_dirs, label_maps), each keyed 'train' / 'valid' / 'test'.
    """
    fold_dirs = {}
    label_maps = {}
    for split in ('train', 'valid', 'test'):
        split_dir = os.path.join(save_dataset_dir, f'{split}_fold_{fold}')
        susceptible_dir = os.path.join(split_dir, '__preS')
        resilient_dir = os.path.join(split_dir, '__preR')
        unified_dir = os.path.join(split_dir, f'__{split}')

        consolidate_files(susceptible_dir, resilient_dir, unified_dir)
        fold_dirs[split] = unified_dir
        label_maps[split] = map_filenames_to_labels(susceptible_dir, resilient_dir)

    return fold_dirs, label_maps


def prepare_datasets_with_scaling(train_dataset, valid_dataset, args):
    """Fit the scaler on train and apply it to train and validation.

    Returns (train_dataset, valid_dataset, train_scaler); the scaler is None when
    --use_standard_scaler is off.
    """
    if not args.use_standard_scaler:
        return train_dataset, valid_dataset, None

    train_scaler = fit_scaler(train_dataset.source_seq, train_dataset.source_mask)
    train_dataset.source_seq = apply_scaler(train_dataset.source_seq, train_scaler)
    valid_dataset.source_seq = apply_scaler(valid_dataset.source_seq, train_scaler)
    return train_dataset, valid_dataset, train_scaler
