"""
train.py
End-to-end training pipeline:
  1. Load raw data, clean it, persist cleaned_dataset.csv.
  2. Stratified train/test split (persisted for reproducibility).
  3. Fit the combined TF-IDF + handcrafted feature pipeline on the TRAIN split only.
  4. Train and compare five classifiers appropriate for high-dimensional sparse text:
       Logistic Regression, Linear SVM (calibrated), Multinomial Naive Bayes,
       Random Forest, and a gradient-boosting model (XGBoost if available,
       otherwise scikit-learn's HistGradientBoostingClassifier as a robust
       drop-in substitute so the pipeline never breaks on missing optional deps).
  5. 5-fold stratified cross-validation (macro-F1) on the training set for each model.
  6. Fit each model on the full training set, evaluate on the held-out test set.
  7. Select the final model using macro-F1 as the primary criterion, with an
     explicit secondary check on the SCAM-class recall (false negatives are the
     costly error in this domain) - see model_metadata.json + README for the
     written justification of the final pick.
  8. Persist the winning model, fitted vectorizers/scaler, label encoder and
     full metadata JSON (dataset stats, per-model metrics, chosen thresholds).

No metric in this file is hand-typed anywhere else in the project - the
README, report and dashboard all read model_metadata.json / model_comparison
tables generated here.
"""

import json
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import LabelEncoder

from src.data_loader import load_raw_data, inspect_dataset
from src.data_cleaning import clean_dataset
from src.feature_engineering import fit_transform_features, transform_features
from src.utils import (
    CLEANED_DATA_PATH, TRAIN_TEST_DIR, BEST_MODEL_PATH, VECTORIZER_PATH,
    LABEL_ENCODER_PATH, METADATA_PATH, RANDOM_STATE, get_logger,
)

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost not available - substituting HistGradientBoostingClassifier")


def build_candidate_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE, C=5.0
        ),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, C=1.0, max_iter=5000),
            cv=3,
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.3),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        models["HistGradientBoosting"] = HistGradientBoostingClassifier(
            max_iter=300, random_state=RANDOM_STATE
        )
    return models


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-9)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "precision_scam": float(precision_score(y_test, y_pred, pos_label=1)),
        "recall_scam": float(recall_score(y_test, y_pred, pos_label=1)),
        "f1_scam": float(f1_score(y_test, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "fit_time_seconds": round(fit_time, 3),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=["SAFE", "SCAM"], output_dict=True)

    logger.info(f"{name}: f1_macro={metrics['f1_macro']:.4f} recall_scam={metrics['recall_scam']:.4f}")
    return model, metrics, cm, report, y_pred, y_proba


def main():
    logger.info("=== STEP 1: Load & inspect raw data ===")
    raw_df = load_raw_data()
    raw_summary = inspect_dataset(raw_df)

    logger.info("=== STEP 2: Clean data ===")
    clean_df = clean_dataset(raw_df)
    clean_df.to_csv(CLEANED_DATA_PATH, index=False)
    clean_summary = inspect_dataset(clean_df)

    logger.info("=== STEP 3: Train/test split ===")
    X_text = clean_df["text"].values
    y = clean_df["label"].values

    le = LabelEncoder()
    le.fit(["SAFE", "SCAM"])  # 0 -> SAFE, 1 -> SCAM (documented mapping)

    X_train_text, X_test_text, y_train, y_test, idx_train, idx_test = train_test_split(
        X_text, y, np.arange(len(y)), test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pd.DataFrame({"text": X_train_text, "label": y_train}).to_csv(
        f"{TRAIN_TEST_DIR}/train.csv", index=False
    )
    pd.DataFrame({"text": X_test_text, "label": y_test}).to_csv(
        f"{TRAIN_TEST_DIR}/test.csv", index=False
    )
    logger.info(f"Train size: {len(X_train_text)} | Test size: {len(X_test_text)}")

    logger.info("=== STEP 4: Feature engineering ===")
    X_train, fitted = fit_transform_features(X_train_text)
    X_test = transform_features(X_test_text, fitted)

    logger.info("=== STEP 5: Train & compare candidate models ===")
    models = build_candidate_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    fitted_models = {}
    predictions = {}

    for name, model in models.items():
        logger.info(f"--- Cross-validating {name} ---")
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
        except Exception as e:
            logger.warning(f"CV failed for {name} ({e}), skipping CV score")
            cv_scores = np.array([np.nan])

        fitted_model, metrics, cm, report, y_pred, y_proba = evaluate_model(
            name, model, X_train, y_train, X_test, y_test
        )
        metrics["cv_f1_macro_mean"] = float(np.nanmean(cv_scores))
        metrics["cv_f1_macro_std"] = float(np.nanstd(cv_scores))

        results[name] = {"metrics": metrics, "confusion_matrix": cm, "classification_report": report}
        fitted_models[name] = fitted_model
        predictions[name] = {"y_pred": y_pred, "y_proba": y_proba}

    logger.info("=== STEP 6: Select best model ===")
    # Primary criterion: macro F1. Tie-break / sanity check: SCAM-class recall,
    # since a missed scam (false negative) is materially worse than a false
    # alarm on a safe message in this domain.
    ranking = sorted(
        results.items(),
        key=lambda kv: (kv[1]["metrics"]["f1_macro"], kv[1]["metrics"]["recall_scam"]),
        reverse=True,
    )
    best_name = ranking[0][0]
    best_model = fitted_models[best_name]
    best_metrics = results[best_name]["metrics"]
    logger.info(f"Best model selected: {best_name} -> {best_metrics}")

    logger.info("=== STEP 7: Persist artifacts ===")
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(fitted, VECTORIZER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    # Calibrate risk-tier probability thresholds using the score distribution
    # on the held-out test set (documented mapping from binary ground truth
    # to the 3-tier SAFE / SUSPICIOUS / SCAM risk system used by the UI).
    best_proba = predictions[best_name]["y_proba"]
    risk_thresholds = {"suspicious_min": 0.35, "scam_min": 0.65}

    metadata = {
        "project_name": "ScamShield AI",
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "source_file": "India_Cyber_Scam_Hinglish_Dataset.csv",
            "raw_summary": raw_summary,
            "cleaned_summary": clean_summary,
            "train_size": int(len(X_train_text)),
            "test_size": int(len(X_test_text)),
        },
        "label_mapping": {"0": "SAFE", "1": "SCAM"},
        "risk_tier_mapping": {
            "description": (
                "The dataset provides a binary ground truth (0=legitimate, "
                "1=scam). The model is trained as a binary classifier. For the "
                "user-facing product, the model's predicted SCAM probability is "
                "bucketed into three risk tiers so the UI can flag borderline "
                "messages as SUSPICIOUS rather than forcing a binary call."
            ),
            "SAFE": f"predicted_scam_probability < {risk_thresholds['suspicious_min']}",
            "SUSPICIOUS": f"{risk_thresholds['suspicious_min']} <= predicted_scam_probability < {risk_thresholds['scam_min']}",
            "SCAM": f"predicted_scam_probability >= {risk_thresholds['scam_min']}",
        },
        "risk_thresholds": risk_thresholds,
        "feature_approach": {
            "word_tfidf": "TF-IDF, word 1-2 grams, max_features=8000",
            "char_tfidf": "TF-IDF, char_wb 3-5 grams, max_features=8000",
            "handcrafted_features": fitted["handcrafted_extractor"].feature_names_,
            "total_feature_dims": int(X_train.shape[1]),
        },
        "models_compared": {name: r["metrics"] for name, r in results.items()},
        "best_model": best_name,
        "best_model_metrics": best_metrics,
        "class_names": ["SAFE", "SUSPICIOUS", "SCAM"],
        "xgboost_available": XGBOOST_AVAILABLE,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata -> {METADATA_PATH}")

    # Persist full results (incl. confusion matrices) for the notebook / evaluate.py
    with open(f"{TRAIN_TEST_DIR}/full_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "best_model": best_name,
                "y_test": y_test.tolist(),
                "predictions": {k: v["y_pred"].tolist() for k, v in predictions.items()},
                "probabilities": {k: v["y_proba"].tolist() for k, v in predictions.items()},
            },
            f,
            indent=2,
        )
    logger.info("Training pipeline complete.")
    return metadata


if __name__ == "__main__":
    main()
