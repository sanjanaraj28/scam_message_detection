"""
data_cleaning.py
Cleans the raw dataset without destroying scam-predictive signal.

Design decision (documented here and in the README):
The uploaded dataset represents transcribed Hinglish phone-call snippets with
a binary ground-truth label (0 = legitimate, 1 = scam) plus metadata columns
(scam_category, caller_type, audio_duration, urgency_level,
contains_blackmail, language_style) that describe *how the call was
categorised after the fact*. Those metadata columns are NOT available for a
real user pasting an arbitrary text message into the app, so they are kept
only for EDA / error-analysis and are NEVER used as model input features.
The deployed model uses the `text` column only.
"""

import re
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{3,}")
MULTI_SPACE_PATTERN = re.compile(r"\s+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def basic_text_clean(text: str) -> str:
    """
    Light-touch normalisation that preserves scam-relevant signal
    (URLs, phone numbers, currency symbols, punctuation, capitalisation
    ratios are all measured *before* this step, see feature_engineering.py).
    This function only fixes whitespace/HTML noise; it does NOT strip
    digits, symbols or lowercase everything, because that would destroy
    features that are engineered downstream.
    """
    if not isinstance(text, str):
        return ""
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = text.strip()
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    return text


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      1. Drop exact duplicate rows.
      2. Drop rows with empty/whitespace-only text.
      3. Normalise whitespace / strip stray HTML.
      4. Drop the constant `language_style` column (100% 'hinglish' -> zero
         information for modelling).
      5. Reset index.
    """
    n_before = len(df)

    df = df.drop_duplicates().copy()
    n_after_dupes = len(df)
    logger.info(f"Removed {n_before - n_after_dupes} exact duplicate rows")

    df["text"] = df["text"].apply(basic_text_clean)
    df = df[df["text"].str.len() > 0].copy()
    n_after_empty = len(df)
    logger.info(f"Removed {n_after_dupes - n_after_empty} empty/whitespace-only messages")

    # -----------------------------------------------------------------
    # IMPORTANT DATA-QUALITY FINDING (documented in README / report / notebook):
    # This dataset is template-generated: the 9,622 rows above collapse to
    # only ~743 DISTINCT message strings, each repeated many times with
    # different simulated metadata (caller_type, audio_duration, etc.) to
    # inflate the row count to 10,000. If an identical message string is
    # allowed to appear in BOTH the train and test split, the model gets to
    # memorise exact strings rather than generalise, which silently inflates
    # every reported metric to ~100%. To keep results honest we deduplicate
    # down to one row per distinct message text before splitting.
    # -----------------------------------------------------------------
    n_before_text_dedup = len(df)
    df = df.drop_duplicates(subset=["text"]).copy()
    n_after_text_dedup = len(df)
    logger.info(
        f"Removed {n_before_text_dedup - n_after_text_dedup} rows that were repeats of an "
        f"already-seen message TEXT (kept 1 row per distinct message to prevent train/test leakage)"
    )

    if "language_style" in df.columns and df["language_style"].nunique() <= 1:
        logger.info("Dropping constant column 'language_style' (no predictive value)")
        df = df.drop(columns=["language_style"])

    df = df.reset_index(drop=True)
    logger.info(f"Cleaning complete: {n_before} -> {len(df)} rows")
    return df


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.utils import CLEANED_DATA_PATH

    raw = load_raw_data()
    cleaned = clean_dataset(raw)
    cleaned.to_csv(CLEANED_DATA_PATH, index=False)
    logger.info(f"Saved cleaned dataset -> {CLEANED_DATA_PATH}")
