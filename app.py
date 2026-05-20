"""
FastAPI Web Interface
=====================
Simple web app for the Amharic NER system.
- GET  /     → HTML form for entering Amharic text
- POST /predict → Returns detected entities as JSON
"""

import os
import logging

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from inference.predictor import NERPredictor

logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title="Amharic NER System",
    description="Named Entity Recognition for Amharic using XLM-RoBERTa",
    version="1.0.0",
)

# Templates directory
templates = Jinja2Templates(directory="templates")

# ── Model loading ────────────────────────────────────────────
# The predictor is initialized lazily on first request
_predictor = None

MODEL_PATH = os.environ.get("NER_MODEL_PATH", "outputs/checkpoints/best_model")


def get_predictor() -> NERPredictor:
    """Load the model once and cache it."""
    global _predictor
    if _predictor is None:
        logger.info("Loading NER model from %s", MODEL_PATH)
        _predictor = NERPredictor(MODEL_PATH)
    return _predictor


# ── Routes ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page with the input form."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(text: str = Form(...)):
    """
    Predict named entities in the submitted Amharic text.

    Args:
        text: Amharic text from the form.

    Returns:
        JSON with the input text and list of detected entities.
    """
    predictor = get_predictor()
    entities = predictor.predict(text)

    return JSONResponse(content={
        "text": text,
        "entities": entities,
    })
