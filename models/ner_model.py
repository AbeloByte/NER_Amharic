"""
NER Model
=========
Wrapper around XLM-RoBERTa for token classification (NER).
Provides a clean interface to load the pretrained model and
configure it with custom NER labels.
"""

import logging

from transformers import XLMRobertaForTokenClassification

logger = logging.getLogger(__name__)


def build_model(
    model_name: str,
    num_labels: int,
    id2label: dict,
    label2id: dict,
) -> XLMRobertaForTokenClassification:
    """
    Build an XLM-RoBERTa model for token classification.

    Loads pretrained weights and adds a linear classification head
    whose output size matches the number of NER labels.

    Args:
        model_name: HuggingFace model identifier (e.g., "xlm-roberta-base").
        num_labels: Number of distinct NER labels.
        id2label:   Mapping from integer id → label string.
        label2id:   Mapping from label string → integer id.

    Returns:
        Configured XLMRobertaForTokenClassification model.
    """
    model = XLMRobertaForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    logger.info(
        "Loaded %s with %d labels: %s",
        model_name,
        num_labels,
        list(label2id.keys()),
    )
    return model
