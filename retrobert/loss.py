import torch
import torch.nn as nn
import torch.nn.functional as F

class F1_Loss(nn.Module):
    """
    Calculate F1 score for multilabel classification.
    Can work with GPU tensors and handles 4 classes.

    Attributes:
        epsilon (float): A small number to prevent division by zero.
    """
    def __init__(self, epsilon=1e-7):
        super(F1_Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        batch_size, num_classes = y_pred.shape
        y_pred = y_pred.reshape(-1, num_classes)
        y_true = y_true.reshape(-1)

        # Convert labels to one-hot encoding
        y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).to(torch.float32)
        y_pred_probs = F.softmax(y_pred, dim=1)

        # Calculate True Positives, False Positives, and False Negatives
        tp = (y_true_one_hot * y_pred_probs).sum(dim=0)
        fn = (y_true_one_hot * (1 - y_pred_probs)).sum(dim=0)
        fp = ((1 - y_true_one_hot) * y_pred_probs).sum(dim=0)

        # Precision, Recall, and F1 Score
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)

        return 1 - f1.mean()