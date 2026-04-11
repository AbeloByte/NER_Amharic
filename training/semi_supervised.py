"""
Semi-Supervised Learning Pipeline (Self-Training)
==================================================
Implements iterative pseudo-labeling:
1. Train initial model on labeled data
2. Predict labels on unlabeled data
3. Filter by confidence threshold
4. Merge high-confidence pseudo-labels with training data
5. Retrain the model
6. Repeat for N iterations
"""

import logging
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from configs.config import NERConfig
from data.data_loader import NERDataset, load_unlabeled
from models.ner_model import build_model
from preprocessing.tokenization import get_tokenizer
from training.trainer import train_model

logger = logging.getLogger(__name__)


def pseudo_label_sentences(
    model: torch.nn.Module,
    sentences: List[str],
    tokenizer,
    config: NERConfig,
    device: torch.device,
) -> List[Tuple[List[str], List[str]]]:
    """
    Generate pseudo-labels for unlabeled sentences.

    For each sentence:
    - Tokenize and run inference
    - Compute softmax probabilities
    - If ALL tokens exceed the confidence threshold, accept the sentence
    - Map predictions back to word-level labels

    Args:
        model:     Trained NER model.
        sentences: List of raw Amharic sentence strings.
        tokenizer: XLM-RoBERTa tokenizer.
        config:    NERConfig with threshold and label mappings.
        device:    Torch device.

    Returns:
        List of (tokens, pseudo_labels) for accepted sentences.
    """
    model.eval()
    accepted = []

    for sentence in sentences:
        # Split sentence into word tokens
        tokens = sentence.split()
        if not tokens:
            continue

        # Tokenize
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids()

        # Run inference
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (1, seq_len, num_labels)

        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1)  # (1, seq_len, num_labels)
        max_probs, pred_ids = probs.max(dim=-1)  # (1, seq_len)
        max_probs = max_probs.squeeze(0)
        pred_ids = pred_ids.squeeze(0)

        # Map sub-token predictions back to word-level
        word_labels = []
        word_confidences = []
        previous_word_id = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != previous_word_id:
                # First sub-token of each word
                label = config.id2label[pred_ids[idx].item()]
                confidence = max_probs[idx].item()
                word_labels.append(label)
                word_confidences.append(confidence)
            previous_word_id = word_id

        # Check that we got labels for all tokens
        if len(word_labels) != len(tokens):
            continue

        # Accept only if ALL token confidences exceed the threshold
        min_confidence = min(word_confidences) if word_confidences else 0.0
        if min_confidence >= config.confidence_threshold:
            accepted.append((tokens, word_labels))

    logger.info(
        "Pseudo-labeling: %d / %d sentences accepted (threshold=%.2f)",
        len(accepted),
        len(sentences),
        config.confidence_threshold,
    )
    return accepted


def self_training_pipeline(
    labeled_sentences: List[Tuple[List[str], List[str]]],
    unlabeled_path: str,
    config: NERConfig,
    device: torch.device,
    val_loader=None,
) -> torch.nn.Module:
    """
    Run the full self-training loop.

    Args:
        labeled_sentences: Initial labeled data as (tokens, labels) tuples.
        unlabeled_path:    Path to the unlabeled text file.
        config:            NERConfig instance.
        device:            Torch device.
        val_loader:        Optional validation DataLoader.

    Returns:
        The final trained model after all self-training iterations.
    """
    tokenizer = get_tokenizer(config.model_name)
    unlabeled_sentences = load_unlabeled(unlabeled_path)
    current_training_data = list(labeled_sentences)

    # Build initial model from pretrained base weights (only once)
    model = build_model(
        config.model_name,
        config.num_labels,
        config.id2label,
        config.label2id,
    )
    model.to(device)

    for iteration in range(1, config.self_training_iterations + 1):
        logger.info("=" * 60)
        logger.info("Self-training iteration %d / %d", iteration, config.self_training_iterations)
        logger.info("Training set size: %d sentences", len(current_training_data))
        logger.info("=" * 60)

        # ── Create dataset & dataloader ──────────────────────
        train_dataset = NERDataset(
            current_training_data, tokenizer, config.label2id, config.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )

        # ── Train (reuses the model from the previous iteration) ─
        model = train_model(model, train_loader, val_loader, config, device)

        # ── Pseudo-label unlabeled data ──────────────────────
        if not unlabeled_sentences:
            logger.info("No unlabeled sentences remaining. Stopping.")
            break

        pseudo_labeled = pseudo_label_sentences(
            model, unlabeled_sentences, tokenizer, config, device
        )

        if not pseudo_labeled:
            logger.info("No new pseudo-labeled data accepted. Stopping early.")
            break

        # Merge pseudo-labels into training set
        current_training_data.extend(pseudo_labeled)
        logger.info(
            "Added %d pseudo-labeled sentences. New training set: %d",
            len(pseudo_labeled),
            len(current_training_data),
        )

        # Remove accepted sentences from unlabeled pool to avoid duplicates
        accepted_texts = {" ".join(tokens) for tokens, _ in pseudo_labeled}
        unlabeled_sentences = [
            s for s in unlabeled_sentences if s not in accepted_texts
        ]

    logger.info("Self-training complete.")
    return model
