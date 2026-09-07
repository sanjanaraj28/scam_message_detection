"""
predict.py
Inference-time interface. Loads the exact artifacts saved by train.py
(model, vectorizers, scaler, metadata) once, then exposes `predict_message()`
which the Flask app (and the automated tests) call for every user-submitted
message. Uses the identical feature pipeline as training - no skew.
"""

import json
import joblib
import numpy as np

from src.feature_engineering import transform_features, detect_indicators
from src.utils import (
    BEST_MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH, METADATA_PATH, get_logger,
)

logger = get_logger(__name__)

MAX_MESSAGE_LENGTH = 3000

RECOMMENDATIONS = {
    "SAFE": "No action needed. The message does not show signs of a scam, but always stay cautious with unsolicited requests for money or personal information.",
    "SUSPICIOUS": "Be cautious. Do not click on links, do not share OTP/PIN/passwords, and independently verify the sender through an official channel before responding.",
    "SCAM": "Do not click any links or share OTP, PIN, passwords, or banking details. Do not make any payment. Block the sender and report the message to the National Cyber Crime Helpline (1930) or cybercrime.gov.in.",
}


class ScamPredictor:
    """Loads model artifacts once and serves predictions."""

    def __init__(self):
        self.model = joblib.load(BEST_MODEL_PATH)
        self.fitted = joblib.load(VECTORIZER_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        with open(METADATA_PATH, encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.thresholds = self.metadata["risk_thresholds"]
        logger.info(f"ScamPredictor ready (model={self.metadata['best_model']})")

    def _risk_tier(self, scam_probability: float) -> str:
        if scam_probability >= self.thresholds["scam_min"]:
            return "SCAM"
        if scam_probability >= self.thresholds["suspicious_min"]:
            return "SUSPICIOUS"
        return "SAFE"

    def predict(self, message: str) -> dict:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message must be a non-empty string.")
        message = message[:MAX_MESSAGE_LENGTH]

        X = transform_features([message], self.fitted)
        if hasattr(self.model, "predict_proba"):
            scam_proba = float(self.model.predict_proba(X)[0, 1])
        else:
            raw = float(self.model.decision_function(X)[0])
            scam_proba = 1 / (1 + np.exp(-raw))

        risk_level = self._risk_tier(scam_proba)
        indicators = detect_indicators(message)
        confidence = scam_proba if risk_level != "SAFE" else (1 - scam_proba)

        return {
            "prediction": risk_level,
            "scam_probability": round(scam_proba, 4),
            "confidence": round(float(confidence), 4),
            "risk_level": risk_level,
            "indicators": indicators,
            "recommendation": RECOMMENDATIONS[risk_level],
            "model_used": self.metadata["best_model"],
        }


_predictor = None


def get_predictor() -> ScamPredictor:
    """Lazy singleton so the (relatively expensive) artifact load happens once."""
    global _predictor
    if _predictor is None:
        _predictor = ScamPredictor()
    return _predictor


def predict_message(message: str) -> dict:
    return get_predictor().predict(message)


if __name__ == "__main__":
    samples = [
        "Hi beta, ghar aa gaya hoon, darwaza khol do.",
        "Aapka KYC pending hai. Account 2 ghante mein block ho jayega. OTP turant share kijiye.",
        "Delhi Police se bol raha hoon. Aapke number ka misuse ho raha hai, video viral karunga agar paisa nahi diya.",
    ]
    for s in samples:
        result = predict_message(s)
        print(f"\nMESSAGE: {s}\nRESULT: {json.dumps(result, indent=2)}")
