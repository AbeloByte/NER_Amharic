"""
Training Pipeline
=================
Supervised training loop for the NER model, including:
- Single-epoch training
- Evaluation with entity-level metrics
- Full training loop with checkpointing and logging
"""

import os
import logging
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Run one training epoch.

    Args:
        model:         The NER model.
        dataloader:    Training DataLoader.
        optimizer:     Optimizer (AdamW).
        scheduler:     Learning-rate scheduler.
        device:        Device (CPU/GPU).
        max_grad_norm: Maximum gradient norm for clipping.

    Returns:
        Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # Move batch tensors to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    id2label: dict,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the model on a dataset and compute entity-level metrics.

    Args:
        model:      The NER model.
        dataloader: Evaluation DataLoader.
        id2label:   Mapping from integer id → label string.
        device:     Device (CPU/GPU).

    Returns:
        Dictionary with precision, recall, f1, and accuracy.
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Convert to label strings, ignoring -100 positions
            for pred_seq, label_seq in zip(predictions, labels):
                pred_labels = []
                true_labels = []
                for p, l in zip(pred_seq, label_seq):
                    if l.item() != -100:
                        pred_labels.append(id2label[p.item()])
                        true_labels.append(id2label[l.item()])
                all_predictions.append(pred_labels)
                all_labels.append(true_labels)

    metrics = compute_metrics(all_labels, all_predictions)
    return metrics


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config,
    device: torch.device,
) -> torch.nn.Module:
    """
    Full training loop with logging and checkpointing.

    Args:
        model:        The NER model.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader (optional).
        config:       NERConfig instance.
        device:       Device (CPU/GPU).

    Returns:
        The trained model.
    """
    model.to(device)

    # ── Optimizer & scheduler ────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    best_f1 = 0.0
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ── Training loop ────────────────────────────────────────
    for epoch in range(1, config.num_epochs + 1):
        avg_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, config.max_grad_norm
        )
        logger.info("Epoch %d/%d — Loss: %.4f", epoch, config.num_epochs, avg_loss)

        # Evaluate if validation set is provided
        if val_loader is not None and len(val_loader) > 0:
            metrics = evaluate(model, val_loader, config.id2label, device)
            logger.info(
                "  Val → P: %.4f  R: %.4f  F1: %.4f",
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
            )

            # Save best checkpoint
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_path = os.path.join(config.checkpoint_dir, "best_model")
                model.save_pretrained(best_path)
                logger.info("  ✓ New best F1: %.4f — checkpoint saved", best_f1)
        else:
            logger.warning("  Validation skipped (val_loader is empty or too small)")

        # Save latest checkpoint every epoch
        latest_path = os.path.join(config.checkpoint_dir, "latest_model")
        model.save_pretrained(latest_path)

    logger.info("Training complete. Best F1: %.4f", best_f1)
    return model
