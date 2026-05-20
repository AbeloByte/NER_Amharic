"""
Amharic Text Normalizer
=======================
Handles Amharic-specific text normalization:
- Unicode normalization (NFC)
- Equivalent character normalization (ሀ/ሃ variants, ጸ/ፀ, etc.)
- Whitespace cleanup
- Removal of non-Amharic noise (stray Latin chars, etc.)
"""

import re
import unicodedata


# ── Amharic character equivalence maps ──────────────────────
# Some Amharic characters have multiple Unicode representations
# that are used interchangeably. We normalize to a single form.
CHAR_EQUIVALENTS = {
    "ሃ": "ሀ",  # ha variants
    "ሐ": "ሀ",
    "ኃ": "ሀ",
    "ዐ": "አ",  # a variants
    "ኣ": "አ",
    "ፀ": "ጸ",  # tsa variants
    "ሥ": "ስ",  # si variants
    "ሡ": "ሱ",
    "ሢ": "ሲ",
    "ሣ": "ሳ",
    "ሤ": "ሴ",
    "ሦ": "ሶ",
}


def normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalization to ensure consistent encoding."""
    return unicodedata.normalize("NFC", text)


def normalize_amharic_chars(text: str) -> str:
    """
    Replace equivalent Amharic characters with their canonical form.

    This helps reduce vocabulary sparsity caused by multiple Unicode
    codes representing the same spoken sound.
    """
    for original, replacement in CHAR_EQUIVALENTS.items():
        text = text.replace(original, replacement)
    return text


def remove_noise(text: str) -> str:
    """
    Remove non-Amharic noise while keeping:
    - Amharic characters (Unicode range \\u1200-\\u137F)
    - Amharic punctuation (\\u1360-\\u1368)
    - Digits and basic punctuation
    - Whitespace
    """
    # Keep Amharic, digits, common punctuation, and whitespace
    cleaned = re.sub(
        r"[^\u1200-\u137F\u1360-\u1368\s0-9.,!?;:\-\"]",
        "",
        text,
    )
    return cleaned


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    """
    Full Amharic text normalization pipeline.

    Steps:
    1. Unicode NFC normalization
    2. Amharic character equivalence normalization
    3. Remove non-Amharic noise
    4. Normalize whitespace

    Args:
        text: Raw Amharic text string.

    Returns:
        Cleaned and normalized text.
    """
    text = normalize_unicode(text)
    text = normalize_amharic_chars(text)
    text = remove_noise(text)
    text = normalize_whitespace(text)
    return text
