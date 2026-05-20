"""
Tokenization & Label Alignment
================================
Wraps XLM-RoBERTa tokenizer with NER-specific label alignment.

For token classification, sub-word tokens that are NOT the first piece
of a word receive a label of -100 so the loss function ignores them.
"""

from typing import List, Optional, Tuple

from transformers import XLMRobertaTokenizerFast


def get_tokenizer(model_name: str = "xlm-roberta-base") -> XLMRobertaTokenizerFast:
    """Load the XLM-RoBERTa fast tokenizer."""
    return XLMRobertaTokenizerFast.from_pretrained(model_name)


def tokenize_and_align_labels(
    tokens: List[str],
    labels: List[str],
    tokenizer: XLMRobertaTokenizerFast,
    label2id: dict,
    max_length: int = 128,
) -> dict:
    """
    Tokenize a pre-split sentence and align NER labels to sub-tokens.

    Strategy:
    - The first sub-token of each word keeps the original label.
    - All subsequent sub-tokens of a word get -100 (ignored by CrossEntropy).
    - Special tokens ([CLS], [SEP]) also get -100.

    Args:
        tokens:     List of original word tokens, e.g. ["አበበ", "አዲስ", "አበባ"].
        labels:     Parallel list of BIO labels,    e.g. ["B-PER", "B-LOC", "I-LOC"].
        tokenizer:  XLM-RoBERTa tokenizer.
        label2id:   Mapping from label string to integer.
        max_length: Maximum sequence length (truncate beyond this).

    Returns:
        Dictionary with keys: input_ids, attention_mask, labels (all as lists).
    """
    # Tokenize with word → sub-token mapping
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,  # Return plain lists
    )

    word_ids = encoding.word_ids()  # Maps each sub-token → original word index
    aligned_labels = []

    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            # Special token ([CLS], [SEP], [PAD])
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            # First sub-token of a new word → keep its label
            aligned_labels.append(label2id[labels[word_id]])
        else:
            # Continuation sub-token → ignore in loss
            aligned_labels.append(-100)
        previous_word_id = word_id

    encoding["labels"] = aligned_labels
    return encoding


def tokenize_for_inference(
    tokens: List[str],
    tokenizer: XLMRobertaTokenizerFast,
    max_length: int = 128,
) -> Tuple[dict, List[Optional[int]]]:
    """
    Tokenize pre-split tokens for inference (no labels needed).

    Args:
        tokens:     List of word tokens.
        tokenizer:  XLM-RoBERTa tokenizer.
        max_length: Maximum sequence length.

    Returns:
        Tuple of (encoding dict, word_ids list).
    """
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids()
    return encoding, word_ids
