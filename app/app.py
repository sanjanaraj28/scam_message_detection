"""
app.py
ScamShield AI - Flask web application.

Endpoints:
  GET  /                 -> dashboard UI
  POST /api/predict      -> JSON prediction API
  GET  /api/analytics     -> model/dataset analytics used by the dashboard charts
  GET  /api/examples      -> a few demo messages (clearly not part of training data)

Security notes:
  - Message length is capped server-side (MAX_MESSAGE_LENGTH).
  - User text is only ever rendered back to the client as JSON (jsonify),
    never interpolated into HTML server-side, so there is no server-side
    template-injection / XSS surface from message content.
  - No secrets or API keys are used or stored by this app.
  - DEBUG is off by default; enable only via the FLASK_DEBUG env var for local dev.
"""

import os
import sys
import json
from datetime import datetime

from flask import Flask, request, jsonify, render_template

# Make the project root importable regardless of the current working directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_message, MAX_MESSAGE_LENGTH, get_predictor  # noqa: E402
from src.utils import METADATA_PATH, VIS_DIR  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # 64 KB request body cap

DEMO_EXAMPLES = [
    {"label": "Likely SAFE", "text": "Hi beta, ghar aa gaya hoon, darwaza khol do."},
    {"label": "Likely SUSPICIOUS", "text": "Sir maine aapko email bheja hai, please check karke reply dena aaj hi."},
    {"label": "Likely SCAM", "text": "Aapka KYC pending hai. Account 2 ghante mein block ho jayega. OTP turant share kijiye."},
    {"label": "Likely SCAM", "text": "Delhi Police se bol raha hoon. Aapke number ka misuse ho raha hai, video viral karunga agar paisa nahi diya."},
]


def load_metadata():
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(silent=True) or {}
        message = payload.get("message", "")

        if not isinstance(message, str) or not message.strip():
            return jsonify({"error": "Field 'message' is required and must be non-empty text."}), 400
        if len(message) > MAX_MESSAGE_LENGTH:
            return jsonify({"error": f"Message too long (max {MAX_MESSAGE_LENGTH} characters)."}), 400

        result = predict_message(message)
        result["timestamp"] = datetime.now().isoformat(timespec="seconds")
        result["message_preview"] = (message[:80] + "…") if len(message) > 80 else message
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Internal error while analyzing the message. Please try again."}), 500


@app.route("/api/analytics")
def api_analytics():
    try:
        metadata = load_metadata()
        best = metadata["best_model_metrics"]
        analytics = {
            "dataset_size_raw": metadata["dataset"]["raw_summary"]["n_rows"],
            "dataset_size_cleaned": metadata["dataset"]["cleaned_summary"]["n_rows"],
            "train_size": metadata["dataset"]["train_size"],
            "test_size": metadata["dataset"]["test_size"],
            "best_model": metadata["best_model"],
            "accuracy": best["accuracy"],
            "f1_macro": best["f1_macro"],
            "f1_scam": best["f1_scam"],
            "recall_scam": best["recall_scam"],
            "precision_scam": best["precision_scam"],
            "roc_auc": best["roc_auc"],
            "models_compared": metadata["models_compared"],
            "class_names": metadata["class_names"],
            "risk_thresholds": metadata["risk_thresholds"],
            "training_date": metadata["training_date"],
        }
        return jsonify(analytics), 200
    except FileNotFoundError:
        return jsonify({"error": "Model metadata not found. Run `python run.py --train` first."}), 500


@app.route("/api/examples")
def api_examples():
    return jsonify({"examples": DEMO_EXAMPLES, "note": "Demo examples for UI testing only - not part of the training dataset."})


@app.route("/api/health")
def health():
    try:
        get_predictor()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Request too large."}), 413


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
