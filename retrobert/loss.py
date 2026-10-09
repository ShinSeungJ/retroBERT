"""Loss function: class-weighted soft F1."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class F1_Loss(nn.Module):
    def __init__(self, class_weights=None, epsilon=1e-7):
        super(F1_Loss, self).__init__()
        self.epsilon = epsilon
        if class_weights is not None:
            inverse_proportions = 1.0 / torch.tensor(class_weights, dtype=torch.float32)
            weights = inverse_proportions / inverse_proportions.sum()
            self.class_weights = weights.clone().detach().float()
        else:
            self.class_weights = None

    def forward(self, y_pred, y_true):
        batch_size, num_classes = y_pred.shape
        y_pred = y_pred.reshape(-1, num_classes)
        y_true = y_true.reshape(-1)
        y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).to(torch.float32)
        y_pred_probs = F.softmax(y_pred, dim=1)
        tp = (y_true_one_hot * y_pred_probs).sum(dim=0)
        fn = (y_true_one_hot * (1 - y_pred_probs)).sum(dim=0)
        fp = ((1 - y_true_one_hot) * y_pred_probs).sum(dim=0)
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()
