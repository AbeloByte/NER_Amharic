# 🇪🇹 Amharic NER System

**Named Entity Recognition for Amharic** using XLM-RoBERTa and semi-supervised learning (self-training with pseudo-labeling).

---

## 📁 Project Structure

```
NER/
├── configs/
│   └── config.py              # Hyperparameters & label definitions
├── data/
│   ├── data_loader.py         # CoNLL/text loaders & NERDataset
│   ├── labeled/
│   │   └── sample.conll       # Sample labeled data (BIO format)
│   └── unlabeled/
│       └── sample.txt         # Sample unlabeled Amharic sentences
├── preprocessing/
│   ├── text_normalizer.py     # Amharic-specific text cleaning
│   └── tokenization.py        # XLM-R tokenizer + label alignment
├── models/
│   └── ner_model.py           # XLM-RoBERTa token classification
├── training/
│   ├── trainer.py             # Supervised training loop
│   └── semi_supervised.py     # Self-training pipeline
├── evaluation/
│   └── metrics.py             # Entity-level P/R/F1 (seqeval)
├── inference/
│   └── predictor.py           # Single-text NER prediction
├── utils/
│   └── helpers.py             # Seed, device, save/load utilities
├── templates/
│   └── index.html             # Web UI template
├── app.py                     # FastAPI web server
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

- **Labeled data** goes in `data/labeled/` in CoNLL format (one token + label per line, blank lines between sentences):
  ```
  አበበ B-PER
  አዲስ B-LOC
  አበባ I-LOC
  ውስጥ O
  ይኖራል O
  ። O
  ```
- **Unlabeled data** goes in `data/unlabeled/` as plain text (one sentence per line).

---

## 🏋️ Training

### Supervised Training

Train on labeled data only:

```bash
python main.py train --data data/labeled/sample.conll --epochs 5
```

Options:
- `--epochs N` — number of training epochs (default: 5)
- `--lr 2e-5` — learning rate
- `--batch-size 16` — batch size
- `--output path/to/save` — where to save the model

### Semi-Supervised Training (Self-Training)

Train on labeled + unlabeled data with pseudo-labeling:

```bash
python main.py semi-train \
  --data data/labeled/sample.conll \
  --unlabeled data/unlabeled/sample.txt \
  --iterations 3 \
  --threshold 0.9
```

**How self-training works:**
1. Train initial model on labeled data
2. Predict labels on unlabeled data
3. Accept sentences where **all** tokens have confidence ≥ threshold
4. Add pseudo-labeled sentences to training set
5. Retrain from scratch on expanded dataset
6. Repeat for N iterations

---

## 📊 Evaluation

Evaluate a trained model on labeled test data:

```bash
python main.py evaluate --model outputs/final_model --data data/labeled/sample.conll
```

Outputs entity-level precision, recall, and F1-score.

---

## 🔍 Inference

### Command Line

```bash
python main.py predict --text "አበበ አዲስ አበባ ኖረ"
```

**Expected output:**
```
Input: አበበ አዲስ አበባ ኖረ
Entities:
  አበበ → PER
  አዲስ አበባ → LOC
```

### Web Interface

Start the FastAPI server:

```bash
python main.py serve --port 8000
```

Open `http://localhost:8000` in your browser and enter Amharic text.

---

## ⚙️ Configuration

All hyperparameters are in `configs/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `xlm-roberta-base` | Pretrained model |
| `max_length` | `128` | Max token length |
| `learning_rate` | `2e-5` | Learning rate |
| `batch_size` | `16` | Training batch size |
| `num_epochs` | `5` | Epochs per training run |
| `confidence_threshold` | `0.90` | Pseudo-label acceptance threshold |
| `self_training_iterations` | `3` | Number of self-training rounds |

### Entity Labels (BIO scheme)

| Label | Meaning |
|-------|---------|
| `O` | Outside any entity |
| `B-PER` / `I-PER` | Person name |
| `B-LOC` / `I-LOC` | Location |
| `B-ORG` / `I-ORG` | Organization |

---

## 🧠 Technologies

- **Python** + **PyTorch**
- **Hugging Face Transformers** (XLM-RoBERTa)
- **seqeval** for entity-level evaluation
- **FastAPI** for web interface
- **NumPy / Pandas** for data handling

---

## 📝 License

This project is for educational / academic purposes.
