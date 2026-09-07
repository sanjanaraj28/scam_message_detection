import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import normalize_for_vectorizer
from src.data_cleaning import basic_text_clean, clean_dataset
from src.feature_engineering import HandcraftedFeatureExtractor, detect_indicators
import pandas as pd


def test_normalize_lowercases():
    assert normalize_for_vectorizer("HELLO World") == "hello world"


def test_normalize_collapses_whitespace():
    assert normalize_for_vectorizer("hello    world\n\n") == "hello world"


def test_normalize_handles_none():
    assert normalize_for_vectorizer(None) == ""


def test_basic_text_clean_strips_html():
    assert basic_text_clean("<b>hello</b>  world") == "hello world"


def test_basic_text_clean_non_string():
    assert basic_text_clean(123) == ""


def test_clean_dataset_removes_duplicates_and_dedupes_text():
    df = pd.DataFrame({
        "text": ["hello", "hello", "world", ""],
        "label": [0, 0, 1, 0],
        "language_style": ["hinglish"] * 4,
    })
    cleaned = clean_dataset(df)
    assert len(cleaned) == 2  # "hello" (deduped), "world"; empty string dropped
    assert "" not in cleaned["text"].values
    assert "language_style" not in cleaned.columns


def test_handcrafted_feature_extractor_shape():
    extractor = HandcraftedFeatureExtractor()
    X = extractor.transform(["Hello world!", "Aapka OTP turant share kijiye ₹499"])
    assert X.shape[0] == 2
    assert X.shape[1] == len(extractor.feature_names_)


def test_handcrafted_feature_extractor_detects_url():
    extractor = HandcraftedFeatureExtractor()
    X = extractor.transform(["Visit http://scam.example.com now"])
    url_idx = extractor.feature_names_.index("url_count")
    assert X[0, url_idx] >= 1


def test_detect_indicators_flags_financial_and_urgency():
    text = "Aapka account 2 ghante mein block ho jayega. OTP turant share kijiye."
    indicators = detect_indicators(text)
    assert any("Urgency" in i for i in indicators)
    assert any("Financial" in i for i in indicators)


def test_detect_indicators_empty_for_benign_message():
    indicators = detect_indicators("Hi beta, ghar aa gaya hoon.")
    assert isinstance(indicators, list)


def test_detect_indicators_handles_non_string():
    assert detect_indicators(None) == []
