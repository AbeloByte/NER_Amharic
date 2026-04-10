"""
NER Inference / Predictor
=========================
Takes raw Amharic text, runs it through the trained model,
and returns detected entities with their labels.
"""

import logging
from typing import Dict, List

import torch
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizerFast

from preprocessing.text_normalizer import normalize_text

logger = logging.getLogger(__name__)


class NERPredictor:
    """
    End-to-end NER predictor.

    Usage:
        predictor = NERPredictor("outputs/checkpoints/best_model")
        entities = predictor.predict("አበበ አዲስ አበባ ኖረ")
        # [{"entity": "አበበ", "label": "PER"}, {"entity": "አዲስ አበባ", "label": "LOC"}]
    """

    def __init__(self, model_path: str, max_length: int = 128):
        """
        Load a trained model and tokenizer from disk.

        Args:
            model_path: Path to the saved model directory.
            max_length: Maximum token sequence length.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Load model and tokenizer
        self.model = XLMRobertaForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path)

        # Extract label mappings from the model config
        # HuggingFace saves keys as strings in config.json; convert to int keys
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        logger.info("Predictor loaded from %s with labels: %s", model_path, list(self.id2label.values()))

    def predict(self, text: str) -> List[Dict[str, str]]:
        """
        Predict named entities in Amharic text.

        Args:
            text: Raw Amharic input string.

        Returns:
            List of entity dictionaries with keys:
            - "entity": the entity text
            - "label": entity type (PER, LOC, ORG)

        Example:
            Input:  "አበበ አዲስ አበባ ኖረ"
            Output: [
                {"entity": "አበበ", "label": "PER"},
                {"entity": "አዲስ አበባ", "label": "LOC"},
            ]
        """
        # Normalize the input text
        text = normalize_text(text)
        tokens = text.split()

        if not tokens:
            return []

        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids()

        # Run inference
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).squeeze(0)

        # Map sub-token predictions back to word-level
        word_labels = []
        previous_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != previous_word_id:
                label = self.id2label[predictions[idx].item()]
                word_labels.append(label)
            previous_word_id = word_id

        # Group BIO tags into entities
        entities = self._group_entities(tokens, word_labels)
        return entities

    @staticmethod
    def _group_entities(
        tokens: List[str], labels: List[str]
    ) -> List[Dict[str, str]]:
        """
        Group BIO-tagged tokens into complete entities.

        Merges consecutive B- and I- tags of the same type into
        a single entity span.

        Args:
            tokens: Word-level tokens.
            labels: Corresponding BIO labels.

        Returns:
            List of entity dicts with "entity" and "label" keys.
        """
        entities = []
        current_entity_tokens = []
        current_label = None

        for token, label in zip(tokens, labels):
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity_tokens and current_label:
                    entities.append({
                        "entity": " ".join(current_entity_tokens),
                        "label": current_label,
                    })
                # Start new entity
                current_label = label[2:]  # Remove "B-" prefix
                current_entity_tokens = [token]

            elif label.startswith("I-") and current_label == label[2:]:
                # Continue current entity
                current_entity_tokens.append(token)

            else:
                # "O" tag or label mismatch → flush current entity
                if current_entity_tokens and current_label:
                    entities.append({
                        "entity": " ".join(current_entity_tokens),
                        "label": current_label,
                    })
                current_entity_tokens = []
                current_label = None

        # Flush last entity
        if current_entity_tokens and current_label:
            entities.append({
                "entity": " ".join(current_entity_tokens),
                "label": current_label,
            })

        return entities
