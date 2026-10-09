"""Metric computation.

Nothing here prints. Every function returns numbers (or the raw material for a
table); reporting them is log.py's job.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)


def compute_file_metrics(predictions, filename_label_map):
    """File-level accuracy and F1 from {filename: "Resilient"|"Susceptible"}.

    Also returns the misclassified files, so the caller can report them.
    """
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []
    incorrect_files = []

    for filename, predicted_label in predictions.items():
        predicted_label = 1 if predicted_label == "Resilient" else 0
        predicted_labels.append(predicted_label)
        true_label = filename_label_map.get(filename)
        true_labels.append(true_label)
        if true_label is not None:
            if true_label == predicted_label:
                correct += 1
            else:
                incorrect_files.append((filename, predicted_label, true_label))
        total += 1
    accuracy = correct / total if total > 0 else 0

    f1_class0 = f1_score(true_labels, predicted_labels, pos_label=0)
    f1_class1 = f1_score(true_labels, predicted_labels, pos_label=1)
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')

    return accuracy, f1_class0, f1_class1, macro_f1, incorrect_files


def compute_sequence_metrics(true_labels, pred_labels):
    """Accuracy and per-class precision/recall/F1 over every scored sequence."""
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average=None, zero_division=0)
    recall = recall_score(true_labels, pred_labels, average=None, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average=None)
    return accuracy, precision, recall, f1


def compute_auc_metrics(file_scores):
    """ROC-AUC and average precision for both classes from (true, score) pairs."""
    if len(file_scores) == 0:
        return float('nan'), float('nan'), float('nan')
    y_true = np.array([t for t, _ in file_scores])
    y_score = np.array([s for _, s in file_scores], dtype=float)
    if len(np.unique(y_true)) < 2:
        return float('nan'), float('nan'), float('nan')
    roc = roc_auc_score(y_true, y_score)
    ap_R = average_precision_score(y_true, y_score)
    ap_S = average_precision_score(1 - y_true, -y_score)
    return float(roc), float(ap_S), float(ap_R)


def pooled_auc_metrics(score_folds):
    """AUC over all folds at once, each fold's scores z-scored before pooling."""
    all_true, all_score = [], []
    for fold in score_folds:
        if not fold:
            continue
        tt = np.array([t for t, _ in fold])
        ss = np.array([s for _, s in fold], dtype=float)
        mu, sd = ss.mean(), ss.std()
        ss = (ss - mu) / sd if sd > 0 else ss - mu
        all_true.extend(tt.tolist())
        all_score.extend(ss.tolist())
    return compute_auc_metrics(list(zip(all_true, all_score)))


def pairwise_pooled_auc(score_folds):
    """Concordance over susceptible/resilient pairs compared within their own fold."""
    conc = 0.0
    total = 0
    for fold in score_folds:
        pos = [s for t, s in fold if t == 1]
        neg = [s for t, s in fold if t == 0]
        for ps in pos:
            for ns in neg:
                if ps > ns:
                    conc += 1.0
                elif ps == ns:
                    conc += 0.5
        total += len(pos) * len(neg)
    return conc / total if total > 0 else float('nan')


def pooled_pair_metrics(pairs):
    """Accuracy and macro-F1 over (true, pred) pairs concatenated across folds."""
    if len(pairs) == 0:
        return float('nan'), float('nan')
    true = [t for t, _ in pairs]
    pred = [p for _, p in pairs]
    return accuracy_score(true, pred), f1_score(true, pred, average='macro')


def initialize_kfold_results():
    """Empty accumulator for the per-fold scores."""
    return {
        'file_acc': [], 'file_f1': [],
        'vote_file_acc': [], 'vote_file_f1': [],
        'seq_acc': [], 'seq_f1': [],
        'file_pairs': [], 'vote_pairs': [], 'seq_pairs': [],
        'roc_auc': [], 'ap_S': [], 'ap_R': [], 'score_folds': [],
    }


def collect_fold_results(bucket, file_acc, file_f1,
                         vote_acc, vote_f1, seq_acc, seq_f1, pooled):
    """Add one fold's scores to the accumulator built by initialize_kfold_results()."""
    bucket['file_acc'].append(file_acc)
    bucket['file_f1'].append(file_f1)
    bucket['vote_file_acc'].append(vote_acc)
    bucket['vote_file_f1'].append(vote_f1)
    bucket['seq_acc'].append(seq_acc)
    bucket['seq_f1'].append(seq_f1)
    bucket['file_pairs'].extend(pooled['file'])
    bucket['vote_pairs'].extend(pooled['vote'])
    bucket['seq_pairs'].extend(pooled['seq'])
    bucket['roc_auc'].append(pooled['roc_auc'])
    bucket['ap_S'].append(pooled['ap_S'])
    bucket['ap_R'].append(pooled['ap_R'])
    bucket['score_folds'].append(pooled['score'])


def mean_sd(values):
    """Mean and SAMPLE standard deviation (ddof=1), ignoring NaNs.

    Used to combine per-seed results, where each value is one seed's summary.
    With a single seed the sample SD is undefined and 0.0 is reported so the
    table still renders.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float('nan'), float('nan')
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def seed_summary(bucket):
    """One seed's headline numbers: the mean over its folds."""
    return {
        'acc': float(np.mean(bucket['file_acc'])),
        'f1': float(np.mean(bucket['file_f1'])),
        'auroc': float(np.nanmean(bucket['roc_auc'])),
    }
