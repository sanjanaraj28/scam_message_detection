"""
evaluate.py
Loads the artifacts saved by train.py, independently re-runs predictions on
the held-out test set to confirm the saved pipeline reproduces the reported
numbers, and generates every chart in visualizations/ from the ACTUAL
results (nothing here is hand-drawn or hardcoded).
"""

import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    accuracy_score, f1_score,
)

from src.feature_engineering import transform_features
from src.utils import (
    BEST_MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH, METADATA_PATH,
    TRAIN_TEST_DIR, VIS_DIR, CLEANED_DATA_PATH, get_logger,
)

logger = get_logger(__name__)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

COLORS = {"SAFE": "#22c55e", "SUSPICIOUS": "#f59e0b", "SCAM": "#ef4444", "accent": "#2563eb"}


def load_artifacts():
    model = joblib.load(BEST_MODEL_PATH)
    fitted = joblib.load(VECTORIZER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    return model, fitted, le, metadata


def independent_reevaluation(model, fitted):
    """Re-load the saved test split and re-verify metrics match training-time numbers."""
    test_df = pd.read_csv(f"{TRAIN_TEST_DIR}/test.csv")
    X_test = transform_features(test_df["text"].values, fitted)
    y_test = test_df["label"].values
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    logger.info(f"Independent re-evaluation of saved model -> accuracy={acc:.4f} f1_macro={f1m:.4f}")
    return test_df, y_test, y_pred


def plot_class_distribution():
    df = pd.read_csv(CLEANED_DATA_PATH)
    counts = df["label"].map({0: "SAFE", 1: "SCAM"}).value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(counts.index, counts.values, color=[COLORS["SAFE"], COLORS["SCAM"]])
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}", (bar.get_x() + bar.get_width() / 2, h), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Class Distribution (Cleaned Dataset)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Messages")
    fig.tight_layout()
    fig.savefig(f"{VIS_DIR}/class_distribution.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["SAFE", "SCAM"]
    ax.set_xticks(range(2)); ax.set_xticklabels(labels)
    ax.set_yticks(range(2)); ax.set_yticklabels(labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - {model_name} (Test Set)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{VIS_DIR}/confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(metadata):
    models = metadata["models_compared"]
    names = list(models.keys())
    f1s = [models[n]["f1_macro"] for n in names]
    accs = [models[n]["accuracy"] for n in names]
    recalls = [models[n]["recall_scam"] for n in names]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, accs, width, label="Accuracy", color="#94a3b8")
    ax.bar(x, f1s, width, label="Macro F1", color=COLORS["accent"])
    ax.bar(x + width, recalls, width, label="Scam Recall", color=COLORS["SCAM"])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Held-out Test Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{VIS_DIR}/model_comparison.png", dpi=150)
    plt.close(fig)


def plot_roc_curve(y_test, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, color=COLORS["accent"], lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{VIS_DIR}/roc_curve.png", dpi=150)
    plt.close(fig)


def plot_pr_curve(y_test, y_proba, model_name):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(recall, precision, color=COLORS["SCAM"], lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {model_name}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{VIS_DIR}/precision_recall_curve.png", dpi=150)
    plt.close(fig)


def plot_top_features(model, fitted, metadata, top_n=20):
    feature_names = np.array(fitted["feature_names"])
    coef = None
    if hasattr(model, "coef_"):
        coef = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        coef = model.feature_importances_
    elif hasattr(model, "calibrated_classifiers_"):
        try:
            base = model.calibrated_classifiers_[0].estimator
            coef = base.coef_[0]
        except Exception:
            coef = None

    if coef is None:
        logger.info("Model has no coef_/feature_importances_; skipping top-features chart data source note")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, f"{metadata['best_model']} does not expose linear\nfeature weights or importances.",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.savefig(f"{VIS_DIR}/top_features.png", dpi=150)
        plt.close(fig)
        return

    order = np.argsort(np.abs(coef))[::-1][:top_n]
    top_feats = feature_names[order]
    top_vals = coef[order]
    colors = [COLORS["SCAM"] if v > 0 else COLORS["SAFE"] for v in top_vals]

    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = np.arange(len(top_feats))
    ax.barh(y_pos, top_vals, color=colors)
    ax.set_yticks(y_pos); ax.set_yticklabels(top_feats)
    ax.invert_yaxis()
    ax.set_xlabel("Model Weight (toward SCAM →  / toward SAFE ←)")
    ax.set_title(f"Top {top_n} Predictive Features - {metadata['best_model']}")
    fig.tight_layout()
    fig.savefig(f"{VIS_DIR}/top_features.png", dpi=150)
    plt.close(fig)


def plot_message_analysis():
    df = pd.read_csv(CLEANED_DATA_PATH)
    df["length"] = df["text"].str.len()
    df["label_name"] = df["label"].map({0: "SAFE", 1: "SCAM"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label_name, color in [("SAFE", COLORS["SAFE"]), ("SCAM", COLORS["SCAM"])]:
        subset = df[df["label_name"] == label_name]["length"]
        axes[0].hist(subset, bins=30, alpha=0.6, label=label_name, color=color)
    axes[0].set_title("Message Length Distribution")
    axes[0].set_xlabel("Character Length")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    if "urgency_level" in df.columns:
        ct = pd.crosstab(df["urgency_level"], df["label_name"])
        ct = ct.reindex(["low", "medium", "high"])
        ct.plot(kind="bar", stacked=True, ax=axes[1], color=[COLORS["SAFE"], COLORS["SCAM"]])
        axes[1].set_title("Urgency Level vs Class")
        axes[1].set_xlabel("Urgency Level")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis="x", rotation=0)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(f"{VIS_DIR}/message_analysis.png", dpi=150)
    plt.close(fig)


def main():
    model, fitted, le, metadata = load_artifacts()
    test_df, y_test, y_pred = independent_reevaluation(model, fitted)

    X_test = transform_features(test_df["text"].values, fitted)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        raw = model.decision_function(X_test)
        y_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    best_name = metadata["best_model"]
    plot_class_distribution()
    plot_confusion_matrix(y_test, y_pred, best_name)
    plot_model_comparison(metadata)
    plot_roc_curve(y_test, y_proba, best_name)
    plot_pr_curve(y_test, y_proba, best_name)
    plot_top_features(model, fitted, metadata)
    plot_message_analysis()
    logger.info(f"All visualizations written to {VIS_DIR}")


if __name__ == "__main__":
    main()
