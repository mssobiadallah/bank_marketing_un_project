"""
test_inference.py — Unit tests for src/inference.py
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from src.inference import (
    predict_batch,
    predict_single,
    risk_level,
    validate_input_schema,
)
from src.config import MODELS_DIR, REALISTIC_MODEL_FILE, PREPROCESSING_PIPELINE_B_FILE, OPTIMAL_THRESHOLD


# ---------------------------------------------------------------------------
# Minimal valid input fixture
# ---------------------------------------------------------------------------

VALID_INPUT = {
    "age": 35,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191.0,
}


# ---------------------------------------------------------------------------
# validate_input_schema
# ---------------------------------------------------------------------------

class TestValidateInputSchema:
    def test_returns_empty_list_for_valid_input(self):
        df = pd.DataFrame([VALID_INPUT])
        errors = validate_input_schema(df)
        assert errors == []

    def test_detects_missing_column(self):
        bad = {k: v for k, v in VALID_INPUT.items() if k != "age"}
        df = pd.DataFrame([bad])
        errors = validate_input_schema(df)
        assert any("age" in e for e in errors)

    def test_duration_stripped_with_warning(self):
        with_duration = {**VALID_INPUT, "duration": 300}
        df = pd.DataFrame([with_duration])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            errors = validate_input_schema(df)
            assert any("duration" in str(warning.message) for warning in w)
        assert errors == []

    def test_empty_dataframe_returns_error(self):
        df = pd.DataFrame()
        errors = validate_input_schema(df)
        assert len(errors) > 0

    def test_all_19_required_fields(self):
        """All 19 required input columns must be checked."""
        required = [
            "age", "job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "campaign", "pdays", "previous",
            "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
            "euribor3m", "nr.employed",
        ]
        for col in required:
            bad = {k: v for k, v in VALID_INPUT.items() if k != col}
            df = pd.DataFrame([bad])
            errors = validate_input_schema(df)
            assert any(col in e for e in errors), (
                f"validate_input_schema did not catch missing column: {col}"
            )


# ---------------------------------------------------------------------------
# risk_level
# ---------------------------------------------------------------------------

class TestRiskLevel:
    def test_high_at_0_6(self):
        assert risk_level(0.6) == "High"

    def test_high_above_0_6(self):
        assert risk_level(0.9) == "High"

    def test_medium_at_0_3(self):
        assert risk_level(0.3) == "Medium"

    def test_medium_between_0_3_and_0_6(self):
        assert risk_level(0.45) == "Medium"

    def test_low_below_0_3(self):
        assert risk_level(0.1) == "Low"

    def test_low_at_zero(self):
        assert risk_level(0.0) == "Low"


# ---------------------------------------------------------------------------
# OPTIMAL_THRESHOLD value check
# ---------------------------------------------------------------------------

class TestOptimalThreshold:
    def test_optimal_threshold_is_float(self):
        assert isinstance(OPTIMAL_THRESHOLD, float)

    def test_optimal_threshold_in_valid_range(self):
        """Threshold must be strictly between 0 and 1."""
        assert 0.0 < OPTIMAL_THRESHOLD < 1.0

    def test_optimal_threshold_lower_than_default(self):
        """Tuned threshold must be below 0.5 to improve minority recall."""
        assert OPTIMAL_THRESHOLD < 0.5


# ---------------------------------------------------------------------------
# predict_single
# ---------------------------------------------------------------------------

class TestPredictSingle:
    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Load the real trained model; skip if not available."""
        model_path = MODELS_DIR / REALISTIC_MODEL_FILE
        pipeline_path = MODELS_DIR / PREPROCESSING_PIPELINE_B_FILE
        if not model_path.exists() or not pipeline_path.exists():
            pytest.skip("Trained model not found — run `make train` first.")
        from src.inference import load_model_and_pipeline
        return load_model_and_pipeline(model_path, pipeline_path)

    def test_returns_required_keys(self, loaded_model):
        model, pipeline = loaded_model
        result = predict_single(model, pipeline, VALID_INPUT)
        required_keys = {"prediction", "subscription_probability", "risk_level",
                         "top_shap_features", "duration_excluded"}
        assert required_keys.issubset(set(result.keys()))

    def test_duration_excluded_is_true(self, loaded_model):
        model, pipeline = loaded_model
        result = predict_single(model, pipeline, VALID_INPUT)
        assert result["duration_excluded"] is True

    def test_probability_in_range(self, loaded_model):
        model, pipeline = loaded_model
        result = predict_single(model, pipeline, VALID_INPUT)
        assert 0.0 <= result["subscription_probability"] <= 1.0

    def test_predicted_class_is_0_or_1(self, loaded_model):
        model, pipeline = loaded_model
        result = predict_single(model, pipeline, VALID_INPUT)
        assert result["prediction"] in (0, 1)

    def test_uses_optimal_threshold_not_half(self, loaded_model):
        """Class should be 1 if probability >= OPTIMAL_THRESHOLD, not 0.5."""
        model, pipeline = loaded_model
        result = predict_single(model, pipeline, VALID_INPUT)
        expected = int(result["subscription_probability"] >= OPTIMAL_THRESHOLD)
        assert result["prediction"] == expected

    def test_duration_stripped_silently(self, loaded_model):
        model, pipeline = loaded_model
        with_duration = {**VALID_INPUT, "duration": 999}
        result = predict_single(model, pipeline, with_duration)
        assert result["prediction"] in (0, 1)


# ---------------------------------------------------------------------------
# predict_batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    @pytest.fixture(scope="class")
    def loaded_model(self):
        model_path = MODELS_DIR / REALISTIC_MODEL_FILE
        pipeline_path = MODELS_DIR / PREPROCESSING_PIPELINE_B_FILE
        if not model_path.exists() or not pipeline_path.exists():
            pytest.skip("Trained model not found — run `make train` first.")
        from src.inference import load_model_and_pipeline
        return load_model_and_pipeline(model_path, pipeline_path)

    def test_returns_dataframe_with_rank(self, loaded_model):
        model, pipeline = loaded_model
        df = pd.DataFrame([VALID_INPUT, {**VALID_INPUT, "age": 55}])
        result = predict_batch(model, pipeline, df)
        assert "rank" in result.columns
        assert "subscription_probability" in result.columns
        assert "predicted_class" in result.columns
        assert "risk_level" in result.columns

    def test_rank_sorted_descending(self, loaded_model):
        model, pipeline = loaded_model
        df = pd.DataFrame([VALID_INPUT] * 5)
        result = predict_batch(model, pipeline, df)
        probs = result["subscription_probability"].tolist()
        assert probs == sorted(probs, reverse=True)

    def test_predicted_class_uses_optimal_threshold(self, loaded_model):
        """predict_batch must use OPTIMAL_THRESHOLD, not 0.5."""
        model, pipeline = loaded_model
        df = pd.DataFrame([VALID_INPUT, {**VALID_INPUT, "age": 55}])
        result = predict_batch(model, pipeline, df)
        expected = (result["subscription_probability"] >= OPTIMAL_THRESHOLD).astype(int)
        assert list(result["predicted_class"].values) == list(expected.values)

    def test_empty_dataframe_raises(self, loaded_model):
        model, pipeline = loaded_model
        with pytest.raises(ValueError):
            predict_batch(model, pipeline, pd.DataFrame())

    def test_duration_column_stripped(self, loaded_model):
        model, pipeline = loaded_model
        df = pd.DataFrame([{**VALID_INPUT, "duration": 300}])
        result = predict_batch(model, pipeline, df)
        assert "duration" not in result.columns
