"""
utils.py
Shared constants, path helpers and small utilities used across the project.
Uses relative paths (anchored on this file's location) so the project works
the same on Windows, macOS and Linux regardless of the current working
directory the scripts are launched from.
"""

import os
import re
import logging

# ---------------------------------------------------------------------------
# Project paths (all derived relative to this file -> works cross-platform)
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "original_dataset.csv")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CLEANED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_dataset.csv")
TRAIN_TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "train_test_data")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

VIS_DIR = os.path.join(PROJECT_ROOT, "visualizations")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

for _d in (PROCESSED_DATA_DIR, TRAIN_TEST_DIR, MODELS_DIR, VIS_DIR, REPORTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Domain knowledge: keyword lists used for engineered features & explainability
# These are heuristic word lists (Hinglish + English) commonly seen in Indian
# cyber-scam call/message transcripts. They power the *explanation* layer,
# which is kept separate from the statistical ML model prediction.
# ---------------------------------------------------------------------------
URGENCY_KEYWORDS = [
    "turant", "abhi", "jaldi", "immediately", "urgent", "right now",
    "2 ghante", "ek ghante", "24 hours", "24 ghante", "expire", "block ho jayega",
    "last warning", "final notice", "aaj hi", "abhi hi", "warna", "nahi to",
]

FINANCIAL_KEYWORDS = [
    "otp", "upi", "pin", "account number", "bank account", "kyc", "cvv",
    "debit card", "credit card", "net banking", "paytm", "gpay", "phonepe",
    "transaction", "refund", "charge", "clearance charge", "fee", "penalty",
    "\u20b9", "rupees", "rs.", "amount", "balance", "transfer karo", "payment",
]

ACCOUNT_SECURITY_KEYWORDS = [
    "verify", "verification", "suspend", "block", "deactivate", "kyc pending",
    "account block", "sim block", "aadhaar", "pan card", "digital arrest",
    "fir", "warrant", "case", "investigation", "police", "cbi", "income tax",
]

AUTHORITY_IMPERSONATION_KEYWORDS = [
    "police se bol raha", "cbi", "income tax department", "trai", "rbi",
    "delhi police", "cyber cell", "court", "supreme court", "special investigation",
    "customs department", "narcotics",
]

BLACKMAIL_KEYWORDS = [
    "video", "photo", "viral", "izzat", "badnaam", "gang", "misuse",
    "objectionable", "share karunga", "leak", "compromising",
]

SUSPICIOUS_KEYWORD_GROUPS = {
    "urgency": URGENCY_KEYWORDS,
    "financial": FINANCIAL_KEYWORDS,
    "account_security": ACCOUNT_SECURITY_KEYWORDS,
    "authority_impersonation": AUTHORITY_IMPERSONATION_KEYWORDS,
    "blackmail_threat": BLACKMAIL_KEYWORDS,
}

# ---------------------------------------------------------------------------
# Shared regex patterns (used by both data_cleaning.py and feature_engineering.py
# so detection logic stays perfectly consistent between cleaning and inference)
# ---------------------------------------------------------------------------
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-\s]?)?\d{10}\b|\b\d{4}[-\s]\d{3}[-\s]\d{3}\b")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
CURRENCY_PATTERN = re.compile(r"(\u20b9|rs\.?\s?\d|inr\s?\d|\$\s?\d)", re.IGNORECASE)

CLASS_NAMES = ["SAFE", "SUSPICIOUS", "SCAM"]

RANDOM_STATE = 42


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that prints to stdout with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s", "%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
