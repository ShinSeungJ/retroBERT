import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_utils import apply_scaler
from utils.model_utils import load_model
from log import compute_accuracy_and_log_errors

def predict(model, test_loader, args):
    """ Conduct inference with trained model and return predictions """
    model.eval()
    all_probs = []
    prediction = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(args.device)
            mask = batch['mask'].to(args.device)
            outputs = model(inputs, attention_mask=mask)
            logits = outputs.logits
            # logits = outputs
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu().numpy())  # Collect as list of NumPy arrays
            prediction.extend(preds.cpu().numpy())
    all_probs = np.vstack(all_probs)
    return all_probs, prediction

def process_csv_file(model, filename, filepath, test_dataset, scaler, args):
    """Process a single CSV file and predict its overall class using an improved weighted averaging method."""

    origin_x_values = [-0.1164, -0.0182, -0.0645, -0.0937, -0.2240]  
    origin_y_values = [-0.2297, -0.3475, -0.3701, -0.2918, -0.1989]
    
    set_number = int(filename[3]) -1
    origin_x = origin_x_values[set_number]
    origin_y = origin_y_values[set_number]

    df = pd.read_csv(filepath)
    data_array = df.to_numpy(dtype=np.float32)
    data_array = data_array[:, :-3]

    # apply correction
    for i in range(8):
        data_array[:, i*3] -= origin_x
        data_array[:, i*3+1] -= origin_y

    tail_base = data_array[:, 6:9]
    body = data_array[:, 9:12]

    spine_length = np.linalg.norm(body - tail_base, axis=1)
    spine_length = spine_length.reshape(-1, 1)
    data_array /= spine_length

    data_tensor = torch.tensor(data_array, dtype=torch.float32)

    test_dataset.source_seq, test_dataset.source_mask = test_dataset.generate_test_sequences(data_tensor)
    test_dataset.source_seq = apply_scaler(test_dataset.source_seq, scaler)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)
    probabilities, preds = predict(model, test_loader, args)

    # majority voting
    threshold = 0.5
    resilient_probs = probabilities[:, 1]
    vote_predictions = (resilient_probs >= threshold).astype(int)
    predictions_counts = np.bincount(vote_predictions, minlength=2)
    if np.mean(vote_predictions) > 0.5:
        vote_prediction = 1  # Majority are resilient
    else:
        vote_prediction = 0  # Majority are susceptible

    # Calculate weighted average of the predictions considering both classes
    weights = probabilities.sum(axis=1)  # Total confidence scores (should typically be 1 if softmax)
    weighted_scores = (probabilities[:, 1] * weights) - (probabilities[:, 0] * weights)  # Difference weighted by total confidence

    final_prediction = np.sum(weighted_scores) / np.sum(weights)
    final_decision = int(final_prediction >= 0)  # Threshold of 0 means positive if average is non-negative

    return final_decision, final_prediction, weighted_scores, vote_prediction, preds


def test(model, test_dir, test_dataset, filename_label_map, scaler, args):
    """ 
    Evaluate model on test dataset and return comprehensive metrics 
    """
    final_results = {}
    individual_accuracies = {}
    log_seq_accuracy = []

    print("Final Predictions:")
    print("Filename  | Prediction | Confidence | Accuracy")

    for filename in os.listdir(test_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(test_dir, filename)
            final_decision, final_prediction, weighted_scores, vote_prediction, sequence_preds = process_csv_file(model, filename, filepath, test_dataset, scaler, args)
            prediction_result = "Resilient" if final_decision == 1 else "Susceptible"
            final_results[filename] = prediction_result

            true_label = filename_label_map.get(filename, None)
            if true_label is not None:
                if true_label == 1:
                    correct_predictions = np.sum(weighted_scores >= 0)
                else:
                    correct_predictions = np.sum(weighted_scores < 0)
                total_predictions = len(weighted_scores)
                file_accuracy = correct_predictions / total_predictions
                individual_accuracies[filename] = file_accuracy
                log_seq_accuracy.append(file_accuracy)
                confidence = np.abs(final_prediction)

                print(f"{filename}: {prediction_result:<11}|  {confidence*100:6.2f}%   | {file_accuracy*100:6.2f}%")

    test_accuracy, precision, recall, f1_class0, f1_class1, file_macro_f1, incorrect_files = compute_accuracy_and_log_errors(final_results, filename_label_map)
    if incorrect_files:
        print("Incorrectly predicted files:")
        for file, predicted_label, true_label in incorrect_files:
            status = "Resilient" if predicted_label == 1 else "Susceptible"
            actual_status = "Resilient" if true_label == 1 else "Susceptible"
            print(f"{file}: Predicted:{status} label:{actual_status}")
    
    print(f"File Accuracy: {test_accuracy * 100:.2f}%")
    print(f"File - Class 0 F1: {f1_class0:.4f}, Class 1 F1: {f1_class1:.4f}, Macro F1: {file_macro_f1:.4f}")

    return test_accuracy, precision, recall, file_macro_f1

def run_test(model, test_dir, test_dataset, filename_label_map, train_scaler, args):
    """
    Run testing on the best F1 model and collect results for k-fold analysis
    """
    
    print(f"Inference on test set:")
    print(f"Best F1 Model:")
    best_f1_model_path = os.path.join(args.save_model_path, "checkpoint_best_f1.pth.tar")
    best_f1_model = load_model(model, best_f1_model_path)
    file_acc, precision, recall, file_f1 = test(best_f1_model, test_dir, test_dataset, filename_label_map, train_scaler, args)
    
    return file_acc, precision, recall, file_f1
