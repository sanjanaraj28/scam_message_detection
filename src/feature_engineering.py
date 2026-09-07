"""
feature_engineering.py
Builds the full feature representation used by the models:

  1. TF-IDF word n-grams   (1-2 grams over normalised text)
  2. TF-IDF char n-grams   (3-5 grams, catches Hinglish spelling variants,
                            obfuscated words like "0TP", "b1ock" etc.)
  3. Hand-crafted numeric features engineered from the RAW (pre-lowercase)
     cleaned text: length, digit/url/phone/currency counts, punctuation
     ratios and keyword-group hits.

All three blocks are combined into a single sparse feature matrix that is
reused identically by train.py and predict.py (the fitted vectorizers /
scaler are pickled so there is zero train/serve skew).
"""

import re
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

from src.preprocessing import normalize_for_vectorizer
from src.utils import (
    URL_PATTERN as _URL_PATTERN,
    PHONE_PATTERN as _PHONE_PATTERN,
    CURRENCY_PATTERN as _CURRENCY_PATTERN,
    SUSPICIOUS_KEYWORD_GROUPS,
    get_logger,
)

logger = get_logger(__name__)


class HandcraftedFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that turns a list/Series of raw
    (cleaned, but NOT lowercased) message strings into a numeric feature
    matrix. Stateless (no fitting required) other than remembering the
    feature names for explainability.
    """

    def __init__(self):
        self.feature_names_ = self._build_feature_names()

    def _build_feature_names(self):
        names = [
            "char_length",
            "word_count",
            "digit_count",
            "digit_ratio",
            "uppercase_ratio",
            "exclamation_count",
            "question_count",
            "url_count",
            "phone_number_count",
            "currency_mention_count",
            "special_char_count",
            "avg_word_length",
        ]
        names += [f"kw_{group}_count" for group in SUSPICIOUS_KEYWORD_GROUPS]
        return names

    def fit(self, X, y=None):
        return self

    def _extract_row(self, text: str):
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        lower = text.lower()

        char_length = len(text)
        words = text.split()
        word_count = len(words)
        digits = sum(ch.isdigit() for ch in text)
        digit_ratio = digits / char_length if char_length else 0.0
        upper_letters = sum(1 for ch in text if ch.isupper())
        letters = sum(1 for ch in text if ch.isalpha())
        uppercase_ratio = upper_letters / letters if letters else 0.0
        exclam = text.count("!")
        question = text.count("?")
        url_count = len(_URL_PATTERN.findall(text))
        phone_count = len(_PHONE_PATTERN.findall(text))
        currency_count = len(_CURRENCY_PATTERN.findall(text))
        special_chars = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
        avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0.0

        row = [
            char_length, word_count, digits, digit_ratio, uppercase_ratio,
            exclam, question, url_count, phone_count, currency_count,
            special_chars, avg_word_len,
        ]

        for group, keywords in SUSPICIOUS_KEYWORD_GROUPS.items():
            count = sum(1 for kw in keywords if kw in lower)
            row.append(count)

        return row

    def transform(self, X):
        rows = [self._extract_row(t) for t in X]
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


def detect_indicators(text: str) -> list:
    """
    Human-readable explanation indicators for a single message, used by the
    web app to show *why* a message looks suspicious. This is a heuristic
    layer, kept explicitly separate from the statistical model's prediction.
    """
    if not isinstance(text, str):
        return []
    lower = text.lower()
    indicators = []

    if _URL_PATTERN.search(text):
        indicators.append("Contains a URL/link")
    if _PHONE_PATTERN.search(text):
        indicators.append("Contains a phone number")
    if _CURRENCY_PATTERN.search(text):
        indicators.append("Mentions a money amount / currency")
    if any(kw in lower for kw in SUSPICIOUS_KEYWORD_GROUPS["urgency"]):
        indicators.append("Urgency / time-pressure language")
    if any(kw in lower for kw in SUSPICIOUS_KEYWORD_GROUPS["financial"]):
        indicators.append("Financial / banking request (OTP, UPI, account, etc.)")
    if any(kw in lower for kw in SUSPICIOUS_KEYWORD_GROUPS["account_security"]):
        indicators.append("Account verification / KYC / security threat language")
    if any(kw in lower for kw in SUSPICIOUS_KEYWORD_GROUPS["authority_impersonation"]):
        indicators.append("Impersonation of police / government authority")
    if any(kw in lower for kw in SUSPICIOUS_KEYWORD_GROUPS["blackmail_threat"]):
        indicators.append("Blackmail / threat language")
    if text.count("!") >= 2:
        indicators.append("Excessive exclamation marks")
    upper_letters = sum(1 for ch in text if ch.isupper())
    letters = sum(1 for ch in text if ch.isalpha())
    if letters and (upper_letters / letters) > 0.4 and letters > 8:
        indicators.append("Unusually high use of capital letters")

    return indicators


def build_vectorizers():
    """Create (unfitted) word and char TF-IDF vectorizers with tuned settings."""
    word_vectorizer = TfidfVectorizer(
        preprocessor=normalize_for_vectorizer,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=8000,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        preprocessor=normalize_for_vectorizer,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        max_features=8000,
        sublinear_tf=True,
    )
    return word_vectorizer, char_vectorizer


def fit_transform_features(texts):
    """
    Fit word-TFIDF, char-TFIDF and the numeric handcrafted extractor on
    `texts`, then return the combined sparse feature matrix plus the fitted
    transformers (so they can be pickled and reused at inference time).
    """
    word_vec, char_vec = build_vectorizers()
    hc_extractor = HandcraftedFeatureExtractor()
    scaler = MaxAbsScaler()

    word_matrix = word_vec.fit_transform(texts)
    char_matrix = char_vec.fit_transform(texts)
    hc_matrix = hc_extractor.transform(texts)
    hc_matrix_scaled = scaler.fit_transform(hc_matrix)

    combined = sp.hstack([word_matrix, char_matrix, sp.csr_matrix(hc_matrix_scaled)]).tocsr()

    feature_names = (
        list(word_vec.get_feature_names_out())
        + list(char_vec.get_feature_names_out())
        + list(hc_extractor.get_feature_names_out())
    )

    fitted = {
        "word_vectorizer": word_vec,
        "char_vectorizer": char_vec,
        "handcrafted_extractor": hc_extractor,
        "scaler": scaler,
        "feature_names": feature_names,
    }
    logger.info(f"Fitted feature matrix shape: {combined.shape}")
    return combined, fitted


def transform_features(texts, fitted):
    """Apply already-fitted transformers to new texts (used at inference)."""
    word_matrix = fitted["word_vectorizer"].transform(texts)
    char_matrix = fitted["char_vectorizer"].transform(texts)
    hc_matrix = fitted["handcrafted_extractor"].transform(texts)
    hc_matrix_scaled = fitted["scaler"].transform(hc_matrix)
    combined = sp.hstack([word_matrix, char_matrix, sp.csr_matrix(hc_matrix_scaled)]).tocsr()
    return combined
