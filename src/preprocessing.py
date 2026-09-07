"""
preprocessing.py
Prepares cleaned text for vectorisation. This is intentionally lighter than
a typical spam-filter pipeline: it lowercases text for the TF-IDF stage but
does NOT strip digits, punctuation, URLs or currency symbols, because those
are highly predictive of scam content in this domain (see feature_engineering.py
and the EDA notebook for evidence). This exact function is reused by
predict.py at inference time, so there is no train/serve skew.
"""

import re
from src.utils import get_logger

logger = get_logger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_vectorizer(text: str) -> str:
    """
    Normalisation applied immediately before TF-IDF vectorisation:
      - lowercase (Hinglish + English mixed casing is not semantically
        meaningful for word/char n-grams)
      - collapse repeated whitespace
    Digits, punctuation, currency symbols and URLs are deliberately kept.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def preprocess_series(texts):
    """Vectorised convenience wrapper for a pandas Series / list of strings."""
    return [normalize_for_vectorizer(t) for t in texts]
