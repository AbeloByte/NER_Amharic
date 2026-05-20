"""
NER Configuration Module
========================
Central configuration for all hyperparameters, model settings,
and label definitions used throughout the NER pipeline.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class NERConfig:
    """
    Configuration dataclass holding all hyperparameters and settings
    for the Amharic NER system.
    """

    # ── Model ────────────────────────────────────────────────
    model_name: str = r"C:\Users\HP\Documents\xlm-roberta-base"
    max_length: int = 128  # Maximum token sequence length

    # ── Training ─────────────────────────────────────────────
    learning_rate: float = 5e-5
    batch_size: int = 2
    num_epochs: int = 20
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # ── Semi-supervised learning ─────────────────────────────
    confidence_threshold: float = 0.70   # Min probability to accept a pseudo-label
    self_training_iterations: int = 3    # Number of self-training rounds

    # ── Paths ────────────────────────────────────────────────
    labeled_data_path: str = "data/labeled/sample.conll"
    unlabeled_data_path: str = "data/unlabeled/sample.txt"
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"

    # ── NER Labels (BIO scheme) ──────────────────────────────
    labels: List[str] = field(default_factory=lambda: [
        "O",
        "B-PER", "I-PER",
        "B-LOC", "I-LOC",
        "B-ORG", "I-ORG",
    ])

    # ── Misc ─────────────────────────────────────────────────
    seed: int = 42

    # ── Derived helpers ──────────────────────────────────────
    @property
    def num_labels(self) -> int:
        """Number of distinct NER labels."""
        return len(self.labels)

    @property
    def label2id(self) -> dict:
        """Mapping from label string → integer id."""
        return {label: idx for idx, label in enumerate(self.labels)}

    @property
    def id2label(self) -> dict:
        """Mapping from integer id → label string."""
        return {idx: label for idx, label in enumerate(self.labels)}
