import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def compute_accuracy_and_log_errors(predictions, filename_label_map):
    """ Compute the accuracy of predictions against actual labels and log incorrect predictions """
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

    # compute precision, recall, and f1 scores
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1_class0 = f1_score(true_labels, predicted_labels, pos_label=0)
    f1_class1 = f1_score(true_labels, predicted_labels, pos_label=1)
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')

    return accuracy, precision, recall, f1_class0, f1_class1, macro_f1, incorrect_files


def check_distribution(dataset):
    label_count = {0: 0, 1: 0}

    for data_point in dataset:
        if 'target' in data_point:  # Ensure 'target' key exists
            labels = data_point['target']
            if torch.is_tensor(labels):
            # Iterate over each label in the tensor
                if labels.item() in label_count:
                    label_count[labels.item()] += 1
            else:
                print("Labels are not in tensor format.")
        else:
            print(f"Unexpected structure for dataset element: {data_point}")

    total = sum(label_count.values())  # Sum up all counts to get the total
    print(f"Total samples: {total}")
    proportions = []
    for label, count in label_count.items():
        proportion = count / total
        print(f"Class {label}: {count}, Proportion: {proportion:.2f}")
        proportions.append(proportion)

    return proportions


def initialize_kfold_results():
    """
    Initialize result tracking variables for k-fold cross-validation
    
    Returns:
        dict: Dictionary containing empty lists for tracking results
    """
    return {
        'file_accuracy': [],
        'file_precision': [],
        'file_recall': [],
        'file_f1': []
    }


def collect_fold_results(results, file_acc, file_precision, file_recall, file_f1):
    """
    Collect results from a single fold and append to tracking lists
    
    Args:
        results: Dictionary containing result tracking lists
        file_acc: File-level accuracy for this fold
        file_precision: File-level precision for this fold
        file_recall: File-level recall for this fold
        file_f1: File-level F1 for this fold
    """
    results['file_accuracy'].append(file_acc)
    results['file_precision'].append(file_precision)
    results['file_recall'].append(file_recall)
    results['file_f1'].append(file_f1)


def compute_mean_sem(values):
    """
    Compute mean and standard error of the mean (SEM)
    
    Args:
        values: List of numerical values
        
    Returns:
        tuple: (mean, sem)
    """
    import scipy.stats as stats
    mean = np.mean(values)
    sem = stats.sem(values)
    return mean, sem

def print_final_kfold_results(results, num_folds):
    """
    Print final k-fold cross-validation results summary
    
    Args:
        results: Dictionary containing collected results from all folds
        num_folds: Number of folds used in cross-validation
    """
    if results['file_accuracy']:
        file_acc_mean, file_acc_sem = compute_mean_sem(results['file_accuracy'])
        file_f1_mean, file_f1_sem = compute_mean_sem(results['file_f1'])
        
        print(f"k-fold {num_folds} File Accuracy: {file_acc_mean:.3f} ± {file_acc_sem:.3f}")
        print(f"k-fold {num_folds} File F1: {file_f1_mean:.3f} ± {file_f1_sem:.3f}")
    else:
        print("No results collected - all models failed to load/test")

def print_kfold_table(results, num_folds):
    """
    Print k-fold results in a formatted table with metrics and SEM
    
    Args:
        results: Dictionary containing collected results from all folds
        num_folds: Number of folds used in cross-validation
    """
    if not results['file_accuracy']:
        print("No results collected - all models failed to load/test")
        return
    
    # Compute means and SEMs
    file_acc_mean, file_acc_sem = compute_mean_sem(results['file_accuracy'])
    file_prec_mean, file_prec_sem = compute_mean_sem(results['file_precision'])
    file_rec_mean, file_rec_sem = compute_mean_sem(results['file_recall'])
    file_f1_mean, file_f1_sem = compute_mean_sem(results['file_f1'])
    
    # Print table header
    print("\n" + "=" * 60)
    print(f"{'Metric':<17} {'Mean ± SEM':<15} {'Individual Fold Results'}")
    print("=" * 60)
    
    # File Accuracy row
    fold_results_acc = " ".join([f"{acc:.3f}" for acc in results['file_accuracy']])
    print(f"{'Accuracy':<15} {file_acc_mean:.3f} ± {file_acc_sem:.3f}     {fold_results_acc}")
    
    # File Precision row
    fold_results_prec = " ".join([f"{prec:.3f}" for prec in results['file_precision']])
    print(f"{'Precision':<15} {file_prec_mean:.3f} ± {file_prec_sem:.3f}     {fold_results_prec}")
    
    # File Recall row
    fold_results_rec = " ".join([f"{rec:.3f}" for rec in results['file_recall']])
    print(f"{'Recall':<15} {file_rec_mean:.3f} ± {file_rec_sem:.3f}     {fold_results_rec}")
    
    # File F1 row  
    fold_results_f1 = " ".join([f"{f1:.3f}" for f1 in results['file_f1']])
    print(f"{'F1':<15} {file_f1_mean:.3f} ± {file_f1_sem:.3f}     {fold_results_f1}")
    
    print("=" * 60)