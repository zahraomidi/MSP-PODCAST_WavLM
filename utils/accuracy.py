import torch
from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score
import logging

logger = logging.getLogger("speechbrain")

def WUAccuracy(log_probabilities, targets):
    """Computes raw predictions and flattened targets for WA/UA calc."""
    if log_probabilities.ndim == 3:
        log_probabilities = log_probabilities.squeeze(1)

    predicted_labels = torch.argmax(log_probabilities, dim=-1)
    targets = targets.squeeze(-1)

    return predicted_labels.cpu().numpy(), targets.cpu().numpy()

class AccuracyStats:
    """Tracks accuracy stats and computes WA and UA."""
    def __init__(self, num_classes=4):
        self.correct = 0
        self.total = 0
        self.num_classes = num_classes
        self.class_correct = defaultdict(int)
        self.class_total = defaultdict(int)
        self.all_preds = []
        self.all_targets = []

    def append(self, log_probabilities, targets):
        """This function is for updating the stats according to the prediction
        and target in the current batch.

        Arguments
        ---------
        log_probabilities : torch.Tensor
            Predicted log probabilities (batch_size, time, feature).
        targets : torch.Tensor
            Target (batch_size, time).
        length : torch.Tensor
            Length of target (batch_size,).
        """
        preds, targs = WUAccuracy(log_probabilities, targets)
        self.correct += (preds == targs).sum()
        self.total += len(targs)

        for pred, true in zip(preds, targs):
            self.class_total[true] += 1
            if pred == true:
                self.class_correct[true] += 1

        self.all_preds.extend(preds)
        self.all_targets.extend(targs)     

    def summarize(self):
        """Computes the accuracy metric."""
        """Returns (WA, UA)"""
        wa = self.correct / self.total if self.total > 0 else 0.0

        per_class_acc = []
        for cls in range(self.num_classes):
            total = self.class_total[cls]
            correct = self.class_correct[cls]
            acc = correct / total if total > 0 else 0.0
            per_class_acc.append(acc)

        ua = np.mean(per_class_acc) if per_class_acc else 0.0
        return wa, ua

    def summarize_confusion(self, labels=None, log_results=True):
        cm = confusion_matrix(self.all_targets, self.all_preds, labels=labels)
        samples_per_class = cm.sum(axis=1)
        total_samples = cm.sum()
        if log_results:
            logger.info("Confusion Matrix:\n%s", np.array2string(cm))
            logger.info(f"Samples per class: {samples_per_class}")
            logger.info(f"Total test samples: {total_samples}")

        return cm

    def summarize_ua(self, log_results=True):
        if not self.all_targets or not self.all_preds:
            ua = 0.0
        else:
            ua = recall_score(self.all_targets, self.all_preds, average="macro", zero_division=0)
        if log_results:
            logger.info(f"UA (macro recall): {ua:.4f}")
        return ua
    
    def reset(self):
        """Resets all the stats."""
        self.correct = 0
        self.total = 0
        self.class_correct = defaultdict(int)
        self.class_total = defaultdict(int)
        self.all_preds = []
        self.all_targets = []