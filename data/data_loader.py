"""
Data Loader
============
Functions and PyTorch Dataset class for loading labeled (CoNLL)
and unlabeled Amharic NER data.
"""

import logging
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from preprocessing.text_normalizer import normalize_text
from preprocessing.tokenization import get_tokenizer, tokenize_and_align_labels

logger = logging.getLogger(__name__)


# ── CoNLL Loader ─────────────────────────────────────────────

def load_conll(filepath: str) -> List[Tuple[List[str], List[str]]]:
    """
    Load a CoNLL-formatted NER file.

    Expected format (one token per line, blank lines between sentences):
        token1  LABEL1
        token2  LABEL2
        <blank line>
        ...

    Args:
        filepath: Path to the CoNLL file.

    Returns:
        List of (tokens, labels) tuples, one per sentence.
    """
    sentences = []
    tokens, labels = [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                # End of sentence
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
                continue
            parts = line.split()
            if len(parts) >= 2:
                token = normalize_text(parts[0])
                label = parts[-1]
                if token:  # Skip if normalization removed the token entirely
                    tokens.append(token)
                    labels.append(label)

    # Handle file that doesn't end with a blank line
    if tokens:
        sentences.append((tokens, labels))

    logger.info("Loaded %d labeled sentences from %s", len(sentences), filepath)
    return sentences


# ── Unlabeled Loader ─────────────────────────────────────────

def load_unlabeled(filepath: str) -> List[str]:
    """
    Load unlabeled Amharic text (one sentence per line).

    Args:
        filepath: Path to the text file.

    Returns:
        List of normalized sentence strings.
    """
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = normalize_text(line)
            if line:
                sentences.append(line)

    logger.info("Loaded %d unlabeled sentences from %s", len(sentences), filepath)
    return sentences


# ── PyTorch Dataset ──────────────────────────────────────────

class NERDataset(Dataset):
    """
    PyTorch Dataset for token-classification NER.

    Each item is a tokenized + label-aligned sentence ready for the model.
    """

    def __init__(
        self,
        sentences: List[Tuple[List[str], List[str]]],
        tokenizer,
        label2id: dict,
        max_length: int = 128,
    ):
        """
        Args:
            sentences: List of (tokens, labels) tuples.
            tokenizer: XLM-RoBERTa tokenizer.
            label2id:  Mapping from label string to integer.
            max_length: Maximum token sequence length.
        """
        self.encodings = []

        for tokens, labels in sentences:
            encoding = tokenize_and_align_labels(
                tokens, labels, tokenizer, label2id, max_length
            )
            self.encodings.append(encoding)

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict:
        item = self.encodings[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }
