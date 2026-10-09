"""Console reporting: stream teeing, banners, per-file lines and the summary tables.

Everything that writes to stdout lives here. The numbers it prints are computed in
metric.py, so a change to the report never changes a metric and vice versa.
"""

import sys
from contextlib import contextmanager

import numpy as np
import torch

from .metric import mean_sd, pairwise_pooled_auc, pooled_auc_metrics, pooled_pair_metrics

# --- stream plumbing ---------------------------------------------------------

class _Tee:
    """Duplicate every write to several streams (e.g. terminal + per-fold log file).
    Flushes on each write so the log stays current even if a run is interrupted."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def start_tee(path):
    """Begin duplicating stdout into `path`. Returns the handles stop_tee() needs."""
    handle = open(path, 'w')
    previous = sys.stdout
    sys.stdout = _Tee(previous, handle)
    return handle, previous


def stop_tee(handle, previous):
    """Restore stdout and close the log file opened by start_tee()."""
    sys.stdout = previous
    handle.close()


@contextmanager
def tee_to_file(path):
    """start_tee/stop_tee as a block: everything printed inside also goes to `path`."""
    handle, previous = start_tee(path)
    try:
        yield
    finally:
        stop_tee(handle, previous)


def print_saved(label, path):
    print(f"[saved] {label} -> {path}")


# --- banners and section headers ---------------------------------------------

def print_banner(title, width=92, ch='='):
    print(f"\n{ch*width}")
    print(title)
    print(f"{ch*width}")


def print_fold_header(fold, num_folds, test_cohort, save_model_path):
    print_banner(f"FOLD {fold}/{num_folds} | test cohort={test_cohort} | "
                 f"Checkpoints: {save_model_path}", width=100, ch='=')


def print_scheduler_steps(fold, total_training_steps, warmup_steps, warmup_ratio):
    print(
        f"Fold {fold} scheduler steps: total={total_training_steps}, "
        f"warmup={warmup_steps} ({warmup_ratio*100:.2f}%)"
    )


# --- dataset composition -----------------------------------------------------

def check_distribution(dataset):
    label_count = {0: 0, 1: 0}

    for data_point in dataset:
        if 'target' in data_point:
            labels = data_point['target']
            if torch.is_tensor(labels):
                if labels.item() in label_count:
                    label_count[labels.item()] += 1
            else:
                print("Labels are not in tensor format.")
        else:
            print(f"Unexpected structure for dataset element: {data_point}")

    total = sum(label_count.values())
    print(f"Total samples: {total}")
    proportions = []
    for label, count in label_count.items():
        proportion = count / total
        print(f"Class {label}: {count}, Proportion: {proportion:.2f}")
        proportions.append(proportion)

    return proportions


def report_distribution(label, dataset):
    """Print "<label> Dataset Distribution:" then the class breakdown."""
    print(f"{label} Dataset Distribution:")
    return check_distribution(dataset)


# --- per-animal predictions --------------------------------------------------

def print_prediction_header(title, threshold):
    if title:
        print(f"\n{'='*70}")
        print(title)
        print(f"{'='*70}")
    print(f"Final Predictions (threshold={threshold:.4f}):")
    print("Filename  | Prediction | Confidence | Accuracy |    Vote | Raw Score")


def print_prediction_row(filename, prediction_result, confidence, file_accuracy,
                         vote_result, final_prediction):
    print(f"{filename}: {prediction_result:<11}|  {confidence*100:6.2f}%   | "
          f"{file_accuracy*100:6.2f}%  | {vote_result:<11}| {final_prediction:+.5f}")


def print_incorrect_files(incorrect_files):
    if incorrect_files:
        print("Incorrectly predicted files:")
        for file, predicted_label, true_label in incorrect_files:
            status = "Resilient" if predicted_label == 1 else "Susceptible"
            actual_status = "Resilient" if true_label == 1 else "Susceptible"
            print(f"{file}: Predicted:{status} label:{actual_status}")


# --- per-fold metric reports -------------------------------------------------

def print_file_metrics(accuracy, f1_class0, f1_class1, macro_f1, incorrect_files):
    print(f"\n--- File Metrics (Weighted Confidence) ---")
    print_incorrect_files(incorrect_files)
    print(f"File Accuracy: {accuracy * 100:.2f}%")
    print(f"File - Class 0 F1: {f1_class0:.4f}, Class 1 F1: {f1_class1:.4f}, Macro F1: {macro_f1:.4f}")


def print_vote_metrics(accuracy, macro_f1, incorrect_files):
    print(f"\n--- Vote Metrics (Majority Vote @ 0.5) ---")
    print_incorrect_files(incorrect_files)
    print(f"Vote Accuracy : {accuracy * 100:.2f}%")
    print(f"Vote F1       : Macro={macro_f1:.4f}")


def print_class_metrics(precision, recall, f1):
    """One line per class; shared by the validation and test reports."""
    for i, (prec, rec, f1_) in enumerate(zip(precision, recall, f1)):
        print(f"Class {i} - Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1_:.4f}")


def print_sequence_metrics(mean_file_seq_accuracy, accuracy, precision, recall, f1):
    print(f"\n--- Sequence Metrics ---")
    print(f"Sequence accuracy: {mean_file_seq_accuracy*100:.2f}%")
    print_class_metrics(precision, recall, f1)
    print(f"Overall Test - Accuracy: {accuracy:.4f}, F1: {np.mean(f1):.4f}")


# --- training loop ---------------------------------------------------------

def print_train_step(epoch, step, mean_train_loss, lr):
    log = f"epoch: {epoch}, (step: {step}) | "
    log += f"train loss: {mean_train_loss:.10f} | "
    log += f"lr: {lr:.10f}"
    print(log)


def print_validation_metrics(epoch, step, valid_loss, accuracy, precision, recall, f1):
    log = f"epoch: {epoch}, (step: {step})"
    log += f" | valid loss: {valid_loss:.10f}"
    print(log)
    print_class_metrics(precision, recall, f1)
    print(f"Overall Validation - Accuracy: {accuracy:.4f}, F1: {np.mean(f1):.4f}")


def print_early_stop(epoch, total_epochs, epochs_without_improvement,
                     best_metric, best_f1, best_epoch):
    print(f"Early stop at epoch {epoch}/{total_epochs}: neither validation loss nor F1 "
          f"improved for {epochs_without_improvement} epoch(s) "
          f"(best loss {best_metric:.10f}, best F1 {best_f1:.4f} at epoch {best_epoch}).")


def print_file_auc(roc_auc, ap_S, ap_R):
    print(f"\n--- File AUC (threshold-invariant) ---")
    print(f"ROC-AUC: {roc_auc:.4f} | AP(susceptible): {ap_S:.4f} | AP(resilient): {ap_R:.4f}")


# --- end-of-run tables -------------------------------------------------------

def print_hyperparameters(args, loss_name, fold_warmup_steps_log, fold_total_steps_log):
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETERS")
    print(f"{'='*60}")
    print(f"  Loss function        : {loss_name}")
    print(f"  Epochs               : {args.train_epochs}")
    print(f"  Batch size           : {args.batch_size}")
    print(f"  Learning rate        : {args.learning_rate}")
    print(f"  Weight decay         : {args.weight_decay}")
    print(f"  Warmup percent       : {args.warmup_ratio*100:.2f}%")
    print(f"  Warmup steps (fold)  : {fold_warmup_steps_log}")
    print(f"  Total steps (fold)   : {fold_total_steps_log}")
    print(f"  Grad accum steps     : {args.gradient_accumulation_steps}")
    print(f"  Max seq length       : {args.max_seq_length}")
    print(f"  Input dim            : {args.input_dim}")
    print(f"  Cohort               : {args.cohort} ({args.data_dir})")
    print(f"  Max grad norm        : {args.max_grad_norm}")
    print(f"  Report every step    : {args.report_every_step}")
    print(f"  Eval every step      : {args.eval_every_step}")
    print(f"  Seed                 : {args.seed}")
    print(f"  Spine scale          : {args.spine_scale}")
    print(f"  Standard scaler      : {args.use_standard_scaler}")
    print(f"  Train/val ratio      : {args.train_val_ratio}")
    shuffle = ('none (real experiment)' if args.shuffle == 'none'
               else f"{args.shuffle} (seed={args.shuffle_seed})")
    print(f"  Shuffle control      : {shuffle}")
    print(f"  Early stop patience  : {args.early_stop_patience if args.early_stop_patience > 0 else 'disabled'}")


def print_summary(bucket, num_folds, width=64):
    """Accuracy, macro-F1 and AUROC, mean +/- std across folds."""
    print(f"\n{'='*width}")
    print(f"K-FOLD {num_folds} SUMMARY (mean +/- std across folds)")
    print(f"{'='*width}")
    print(f"{'Model':<14} {'Acc':<15} {'F1':<15} {'AUROC'}")
    print(f"{'-'*width}")
    if bucket['file_acc']:
        acc = f"{np.mean(bucket['file_acc'])*100:.2f}±{np.std(bucket['file_acc'])*100:.2f}"
        f1 = f"{np.mean(bucket['file_f1'])*100:.2f}±{np.std(bucket['file_f1'])*100:.2f}"
        auroc = f"{np.nanmean(bucket['roc_auc']):.3f}±{np.nanstd(bucket['roc_auc']):.3f}"
        print(f"{'Best Model':<14} {acc:<15} {f1:<15} {auroc}")
    print(f"{'='*width}")


def print_aggregate(records, width=64):
    """Mean +/- sample SD across seeds, from the per-seed summary_metrics.json files."""
    seeds = [r['seed'] for r in records]
    first = records[0]
    acc = mean_sd([r['seed_mean']['acc'] for r in records])
    f1 = mean_sd([r['seed_mean']['f1'] for r in records])
    auroc = mean_sd([r['seed_mean']['auroc'] for r in records])

    print(f"\n{'='*width}")
    print(f"AGGREGATE OVER {len(records)} SEED(S) - {first['exp_base']}")
    print(f"{'='*width}")
    print(f"  Cohort               : {first['cohort']}")
    print(f"  Folds per seed       : {first['num_folds']}")
    print(f"  Seeds                : {seeds}")
    print(f"  Shuffle control      : {first['shuffle']}")
    print(f"  +/- is sample SD (ddof=1) across seeds")
    print(f"{'-'*width}")
    print(f"{'Model':<14} {'Acc':<15} {'F1':<15} {'AUROC'}")
    print(f"{'-'*width}")
    print(f"{'Best Model':<14} "
          f"{f'{acc[0]*100:.2f}±{acc[1]*100:.2f}':<15} "
          f"{f'{f1[0]*100:.2f}±{f1[1]*100:.2f}':<15} "
          f"{f'{auroc[0]:.3f}±{auroc[1]:.3f}'}")
    print(f"{'='*width}")
