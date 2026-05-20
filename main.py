"""
Amharic NER System — Main Entry Point
======================================
CLI interface with subcommands:
  - train        : Supervised training on labeled data
  - semi-train   : Self-training with pseudo-labeling
  - evaluate     : Evaluate a trained model
  - predict      : Run inference on a single text
  - serve        : Start the FastAPI web server
"""

import argparse
import logging
import os
import sys

from configs.config import NERConfig
from utils.helpers import set_seed, get_device, setup_logging, save_model


def cmd_train(args):
    """Train the model using supervised learning on labeled data."""
    from torch.utils.data import DataLoader, random_split

    from data.data_loader import load_conll, NERDataset
    from models.ner_model import build_model
    from preprocessing.tokenization import get_tokenizer
    from training.trainer import train_model

    config = NERConfig()
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size

    set_seed(config.seed)
    device = get_device()

    # Load data
    labeled_path = args.data or config.labeled_data_path
    sentences = load_conll(labeled_path)

    if len(sentences) == 0:
        logging.error("No sentences loaded. Check the data file: %s", labeled_path)
        sys.exit(1)

    # Tokenizer
    tokenizer = get_tokenizer(config.model_name)

    # Split into train / validation (80/20)
    split_idx = max(1, int(len(sentences) * 0.8))
    train_sentences = sentences[:split_idx]
    val_sentences = sentences[split_idx:]

    train_dataset = NERDataset(train_sentences, tokenizer, config.label2id, config.max_length)
    val_dataset = NERDataset(val_sentences, tokenizer, config.label2id, config.max_length) if val_sentences else None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size) if val_dataset else None

    # Build model
    model = build_model(config.model_name, config.num_labels, config.id2label, config.label2id)

    # Train
    model = train_model(model, train_loader, val_loader, config, device)

    # Save final model
    output_dir = args.output or os.path.join(config.output_dir, "final_model")
    save_model(model, tokenizer, output_dir)
    logging.info("Training complete. Model saved to %s", output_dir)


def cmd_semi_train(args):
    """Train using self-training / semi-supervised learning."""
    from data.data_loader import load_conll, NERDataset
    from preprocessing.tokenization import get_tokenizer
    from training.semi_supervised import self_training_pipeline

    config = NERConfig()
    if args.epochs:
        config.num_epochs = args.epochs
    if args.threshold:
        config.confidence_threshold = args.threshold
    if args.iterations:
        config.self_training_iterations = args.iterations

    set_seed(config.seed)
    device = get_device()

    # Load labeled data
    labeled_path = args.data or config.labeled_data_path
    labeled_sentences = load_conll(labeled_path)

    # Unlabeled data path
    unlabeled_path = args.unlabeled or config.unlabeled_data_path

    # Run self-training
    model = self_training_pipeline(
        labeled_sentences=labeled_sentences,
        unlabeled_path=unlabeled_path,
        config=config,
        device=device,
    )

    # Save final model
    tokenizer = get_tokenizer(config.model_name)
    output_dir = args.output or os.path.join(config.output_dir, "semi_trained_model")
    save_model(model, tokenizer, output_dir)
    logging.info("Semi-supervised training complete. Model saved to %s", output_dir)


def cmd_evaluate(args):
    """Evaluate a trained model on labeled data."""
    from torch.utils.data import DataLoader

    from data.data_loader import load_conll, NERDataset
    from evaluation.metrics import print_classification_report
    from preprocessing.tokenization import get_tokenizer
    from training.trainer import evaluate
    from utils.helpers import load_model
    from transformers import XLMRobertaForTokenClassification

    config = NERConfig()
    device = get_device()

    # Load model
    model_path = args.model or os.path.join(config.output_dir, "final_model")
    model = load_model(
        XLMRobertaForTokenClassification, model_path,
        config.num_labels, config.id2label, config.label2id,
    )
    model.to(device)

    # Load test data
    tokenizer = get_tokenizer(model_path)
    test_path = args.data or config.labeled_data_path
    test_sentences = load_conll(test_path)

    test_dataset = NERDataset(test_sentences, tokenizer, config.label2id, config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # Evaluate
    metrics = evaluate(model, test_loader, config.id2label, device)
    print(f"\nPrecision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")


def cmd_predict(args):
    """Predict entities in a single Amharic text."""
    config = NERConfig()
    from inference.predictor import NERPredictor

    model_path = args.model or os.path.join(config.output_dir, "final_model")
    predictor = NERPredictor(model_path)

    text = args.text
    entities = predictor.predict(text)

    print(f"\nInput: {text}")
    print("Entities:")
    if entities:
        for ent in entities:
            print(f"  {ent['entity']} → {ent['label']}")
    else:
        print("  No entities detected.")


def cmd_serve(args):
    """Start the FastAPI web server."""
    import uvicorn

    host = args.host or "0.0.0.0"
    port = args.port or 8000

    # Set model path environment variable for the app
    config = NERConfig()
    model_path = args.model or os.path.join(config.output_dir, "final_model")
    os.environ["NER_MODEL_PATH"] = model_path

    print(f"Starting Amharic NER server at http://{host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(
        description="Amharic NER System — XLM-RoBERTa + Semi-Supervised Learning"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── train ────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Supervised training")
    p_train.add_argument("--data", type=str, help="Path to labeled CoNLL file")
    p_train.add_argument("--output", type=str, help="Output directory for saved model")
    p_train.add_argument("--epochs", type=int, help="Number of training epochs")
    p_train.add_argument("--lr", type=float, help="Learning rate")
    p_train.add_argument("--batch-size", type=int, help="Batch size")

    # ── semi-train ───────────────────────────────────────────
    p_semi = subparsers.add_parser("semi-train", help="Self-training (semi-supervised)")
    p_semi.add_argument("--data", type=str, help="Path to labeled CoNLL file")
    p_semi.add_argument("--unlabeled", type=str, help="Path to unlabeled text file")
    p_semi.add_argument("--output", type=str, help="Output directory for saved model")
    p_semi.add_argument("--epochs", type=int, help="Epochs per iteration")
    p_semi.add_argument("--threshold", type=float, help="Confidence threshold (0-1)")
    p_semi.add_argument("--iterations", type=int, help="Number of self-training rounds")

    # ── evaluate ─────────────────────────────────────────────
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    p_eval.add_argument("--model", type=str, help="Path to saved model directory")
    p_eval.add_argument("--data", type=str, help="Path to labeled test data (CoNLL)")

    # ── predict ──────────────────────────────────────────────
    p_pred = subparsers.add_parser("predict", help="Predict entities in text")
    p_pred.add_argument("--text", type=str, required=True, help="Amharic text to analyze")
    p_pred.add_argument("--model", type=str, help="Path to saved model directory")

    # ── serve ────────────────────────────────────────────────
    p_serve = subparsers.add_parser("serve", help="Start the web interface")
    p_serve.add_argument("--model", type=str, help="Path to saved model directory")
    p_serve.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    p_serve.add_argument("--port", type=int, default=8000, help="Server port")

    args = parser.parse_args()
    setup_logging()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "semi-train":
        cmd_semi_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
