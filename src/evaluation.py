from itertools import combinations

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_curve)


def plot_precision_recall_curve(y_true, y_score, path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(12, 6), dpi=80)
    plt.plot(recall, precision)

    plt.yticks([i / 100 for i in range(0, 101, 5)])
    plt.xticks([i / 100 for i in range(0, 101, 5)])

    plt.title('Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.savefig(path)

    return plt


def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr

    eer_idx = np.argmin(np.abs(fnr - fpr))

    return fpr[eer_idx], fnr[eer_idx], thresholds[eer_idx]


def compute_max_f1(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=1)
    f1 = 2 * precision * recall / (precision + recall)

    max_f1_idx = np.argmax(f1)

    return f1[max_f1_idx], precision[max_f1_idx], recall[max_f1_idx], thresholds[max_f1_idx]


def get_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, precision, recall, f1

