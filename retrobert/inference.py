"""Inference, threshold selection and test-set evaluation."""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from .log import (print_banner, print_file_auc, print_file_metrics,
                  print_prediction_header, print_prediction_row, print_sequence_metrics,
                  print_vote_metrics)
from .metric import (collect_fold_results, compute_auc_metrics, compute_file_metrics,
                     compute_sequence_metrics)
from .utils.data_utils import apply_scaler, normalize_spine
from .utils.model_utils import load_model

def predict(model, test_loader, args):
    model.eval()
    all_probs = []
    prediction = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(args.device)
            mask = batch['mask'].to(args.device)
            outputs = model(inputs, attention_mask=mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu().numpy())
            prediction.extend(preds.cpu().numpy())
    all_probs = np.vstack(all_probs)
    return all_probs, prediction

def process_csv_file(model, filename, filepath, test_dataset, scaler, args, threshold=0.0):
    """Process a single pose CSV file and predict its overall class."""
    df = pd.read_csv(filepath)
    data_array = df.to_numpy(dtype=np.float32)

    data_array = normalize_spine(data_array, args.spine_scale)

    data_tensor = torch.tensor(data_array, dtype=torch.float32)

    test_dataset.source_seq, test_dataset.source_mask = test_dataset.generate_test_sequences(data_tensor)
    if scaler is not None:
        test_dataset.source_seq = apply_scaler(test_dataset.source_seq, scaler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)
    probabilities, preds = predict(model, test_loader, args)

    vote_threshold = 0.5
    resilient_probs = probabilities[:, 1]
    vote_predictions = (resilient_probs >= vote_threshold).astype(int)
    predictions_counts = np.bincount(vote_predictions, minlength=2)
    if np.mean(vote_predictions) > 0.5:
        vote_prediction = 1
    else:
        vote_prediction = 0

    weights = probabilities.sum(axis=1)
    weighted_scores = (probabilities[:, 1] * weights) - (probabilities[:, 0] * weights)

    final_prediction = np.sum(weighted_scores) / np.sum(weights)
    final_decision = int(final_prediction >= threshold)

    return final_decision, final_prediction, weighted_scores, vote_prediction, preds

def test(model, test_dir, test_dataset, filename_label_map, scaler, args, threshold=0.0, title=None, show_vote=False):

    final_results = {}
    individual_accuracies = {}
    final_vote = {}
    log_seq_accuracy = []
    all_predictions = []
    all_true_labels = []
    all_filenames = []
    file_scores = []

    print_prediction_header(title, threshold)

    for filename in os.listdir(test_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(test_dir, filename)
            final_decision, final_prediction, weighted_scores, vote_prediction, sequence_preds = process_csv_file(
                model, filename, filepath, test_dataset, scaler, args, threshold=threshold
            )
            prediction_result = "Resilient" if final_decision == 1 else "Susceptible"
            vote_result = "Resilient" if vote_prediction == 1 else "Susceptible"
            final_results[filename] = prediction_result
            final_vote[filename] = vote_result

            true_label = filename_label_map.get(filename, None)
            if true_label is not None:
                file_scores.append((true_label, final_prediction))
                for pred in sequence_preds:
                    all_true_labels.append(true_label)
                    all_filenames.append(filename)
                    all_predictions.append(pred)
                if true_label == 1:
                    correct_predictions = np.sum(weighted_scores >= threshold)
                else:
                    correct_predictions = np.sum(weighted_scores < threshold)
                total_predictions = len(weighted_scores)
                file_accuracy = correct_predictions / total_predictions
                individual_accuracies[filename] = file_accuracy
                log_seq_accuracy.append(file_accuracy)
                confidence = np.abs(final_prediction)
                print_prediction_row(filename, prediction_result, confidence,
                                     file_accuracy, vote_result, final_prediction)

    test_accuracy, f1_class0, f1_class1, file_macro_f1, incorrect_files = compute_file_metrics(
        final_results, filename_label_map)
    print_file_metrics(test_accuracy, f1_class0, f1_class1, file_macro_f1, incorrect_files)

    vote_accuracy, _, _, vote_macro_f1, vote_incorrect = compute_file_metrics(
        final_vote, filename_label_map)
    if show_vote:
        print_vote_metrics(vote_accuracy, vote_macro_f1, vote_incorrect)

    accuracy, precision, recall, f1 = compute_sequence_metrics(all_true_labels, all_predictions)
    print_sequence_metrics(np.mean(log_seq_accuracy), accuracy, precision, recall, f1)
    seq_macro_f1 = float(np.mean(f1))

    file_pairs = []
    for fn, res in final_results.items():
        tl = filename_label_map.get(fn)
        if tl is not None:
            file_pairs.append((tl, 1 if res == "Resilient" else 0))
    vote_pairs = []
    for fn, res in final_vote.items():
        tl = filename_label_map.get(fn)
        if tl is not None:
            vote_pairs.append((tl, 1 if res == "Resilient" else 0))
    seq_pairs = list(zip(all_true_labels, all_predictions))

    fold_roc_auc, fold_ap_S, fold_ap_R = compute_auc_metrics(file_scores)
    print_file_auc(fold_roc_auc, fold_ap_S, fold_ap_R)

    pooled = {'file': file_pairs, 'vote': vote_pairs, 'seq': seq_pairs,
              'score': file_scores, 'roc_auc': fold_roc_auc, 'ap_S': fold_ap_S, 'ap_R': fold_ap_R}

    return test_accuracy, file_macro_f1, vote_accuracy, vote_macro_f1, accuracy, seq_macro_f1, pooled


def run_test(model, fold, fold_dirs, label_maps, test_dataset, scaler, args, results):
    """Score the held-out cohort with the best checkpoint at its untuned threshold."""
    print_banner(f"FOLD {fold} INFERENCE | BEST MODEL", width=100, ch='#')
    best_model = load_model(model, os.path.join(args.save_model_path,
                                                "checkpoint_best_f1.pth.tar"))

    outcome = test(best_model, fold_dirs['test'], test_dataset, label_maps['test'],
                   scaler, args, threshold=0.0,
                   title="Best Model (threshold=0.0)")
    collect_fold_results(results, *outcome)
