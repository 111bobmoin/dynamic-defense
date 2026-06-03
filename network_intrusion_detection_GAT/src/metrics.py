from __future__ import annotations

import torch


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for truth, pred in zip(y_true.view(-1), y_pred.view(-1), strict=True):
        matrix[truth.long(), pred.long()] += 1
    return matrix


def classification_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> dict[str, float | dict[str, float]]:
    matrix = confusion_matrix(y_true.cpu(), y_pred.cpu(), num_classes=num_classes).float()
    total = matrix.sum().item()
    accuracy = matrix.diag().sum().item() / total if total else 0.0

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    f1_scores: list[float] = []
    for class_index in range(num_classes):
        tp = matrix[class_index, class_index].item()
        fp = matrix[:, class_index].sum().item() - tp
        fn = matrix[class_index, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    macro_precision = sum(precision_scores) / num_classes if num_classes else 0.0
    macro_recall = sum(recall_scores) / num_classes if num_classes else 0.0
    macro_f1 = sum(f1_scores) / num_classes if num_classes else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }
