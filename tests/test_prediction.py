import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_message, MAX_MESSAGE_LENGTH


def test_predict_returns_expected_keys():
    result = predict_message("Hi beta, ghar aa gaya hoon, darwaza khol do.")
    for key in ["prediction", "scam_probability", "confidence", "risk_level",
                "indicators", "recommendation", "model_used"]:
        assert key in result


def test_predict_scam_message_flagged_high_risk():
    result = predict_message(
        "Aapka KYC pending hai. Account 2 ghante mein block ho jayega. OTP turant share kijiye."
    )
    assert result["risk_level"] in ("SUSPICIOUS", "SCAM")
    assert result["scam_probability"] > 0.5


def test_predict_safe_message_flagged_low_risk():
    result = predict_message("Hi beta, ghar aa gaya hoon, darwaza khol do.")
    assert result["risk_level"] == "SAFE"
    assert result["scam_probability"] < 0.5


def test_predict_empty_message_raises():
    with pytest.raises(ValueError):
        predict_message("")


def test_predict_whitespace_only_message_raises():
    with pytest.raises(ValueError):
        predict_message("     ")


def test_predict_none_message_raises():
    with pytest.raises(ValueError):
        predict_message(None)


def test_predict_extremely_long_message_is_truncated_not_crashing():
    long_message = "urgent otp block account " * 500
    assert len(long_message) > MAX_MESSAGE_LENGTH
    result = predict_message(long_message)
    assert result["risk_level"] in ("SAFE", "SUSPICIOUS", "SCAM")


def test_predict_probability_is_bounded():
    result = predict_message("Congratulations you have won a lottery prize, click here.")
    assert 0.0 <= result["scam_probability"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
