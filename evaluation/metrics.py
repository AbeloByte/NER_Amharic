"""
Evaluation Metrics
==================
Entity-level NER evaluation using seqeval:
- Precision, Recall, F1-score
- Per-entity-type classification report
"""

import logging
from typing import Dict, List

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> Dict[str, float]:
    """
    Compute entity-level precision, recall, and F1.

    Args:
        true_labels: List of sequences of true BIO labels.
        pred_labels: List of sequences of predicted BIO labels.

    Returns:
        Dictionary with keys: precision, recall, f1.
    """
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_classification_report(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> str:
    """
    Print and return a detailed per-entity classification report.

    Args:
        true_labels: List of sequences of true BIO labels.
        pred_labels: List of sequences of predicted BIO labels.

    Returns:
        Formatted classification report string.
    """
    report = classification_report(true_labels, pred_labels)
    logger.info("\n%s", report)
    return report
