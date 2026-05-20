"""
Utility Helpers
===============
Common utility functions used across the NER pipeline:
seed setting, device selection, model save/load.
"""

import os
import random
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


def get_device() -> torch.device:
    """Return the best available device (CUDA → CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def save_model(model, tokenizer, output_dir: str) -> None:
    """
    Save model weights and tokenizer to disk.

    Args:
        model: The HuggingFace model to save.
        tokenizer: The associated tokenizer.
        output_dir: Directory to write files into.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model and tokenizer saved to %s", output_dir)


def load_model(model_class, output_dir: str, num_labels: int, id2label: dict, label2id: dict):
    """
    Load a previously saved model from disk.

    Args:
        model_class: HuggingFace model class (e.g., XLMRobertaForTokenClassification).
        output_dir: Directory containing saved model files.
        num_labels: Number of NER labels.
        id2label: Mapping from id to label string.
        label2id: Mapping from label string to id.

    Returns:
        The loaded model.
    """
    model = model_class.from_pretrained(
        output_dir,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    logger.info("Model loaded from %s", output_dir)
    return model


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
