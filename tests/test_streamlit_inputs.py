"""tests/test_streamlit_inputs.py — Tests for input validation used by the Streamlit app."""
from __future__ import annotations
import pytest
import pandas as pd

from src.inference import validate_input_schema, _REQUIRED_INPUT_COLS
from src.config import FEATURE_SET_B_COLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_row() -> dict:
    """Minimal valid input dict containing all 19 Set B required columns."""
    return {
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
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 93.2,
        "cons.conf.idx": -42.0,
        "euribor3m": 1.0,
        "nr.employed": 5099.1,
    }


@pytest.fixture()
def valid_df(valid_row) -> pd.DataFrame:
    """5-row DataFrame where all rows are valid."""
    return pd.DataFrame([valid_row] * 5)


# ---------------------------------------------------------------------------
# validate_input_schema
# ---------------------------------------------------------------------------

class TestValidateInputSchema:
    def test_valid_row_returns_no_errors(self, valid_row):
        df = pd.DataFrame([valid_row])
        errors = validate_input_schema(df, feature_set="set_b")
        assert errors == []

    def test_all_19_required_fields_present(self, valid_row):
        """Set B should require exactly 19 raw input columns (no duration)."""
        df = pd.DataFrame([valid_row])
        assert len(valid_row) == len(_REQUIRED_INPUT_COLS)
        errors = validate_input_schema(df, feature_set="set_b")
        assert errors == []

    def test_missing_single_column_returns_error(self, valid_row):
        del valid_row["age"]
        df = pd.DataFrame([valid_row])
        errors = validate_input_schema(df, feature_set="set_b")
        assert any("age" in e for e in errors)

    def test_missing_multiple_columns_reported(self, valid_row):
        del valid_row["age"]
        del valid_row["campaign"]
        del valid_row["euribor3m"]
        df = pd.DataFrame([valid_row])
        errors = validate_input_schema(df, feature_set="set_b")
        missing_mentioned = sum(
            1 for e in errors
            if any(col in e for col in ["age", "campaign", "euribor3m"])
        )
        assert missing_mentioned >= 1  # at least one error mentions missing cols

    def test_duration_column_stripped_silently(self, valid_row):
        """Duration should be stripped without raising, returning no errors."""
        valid_row["duration"] = 300  # inject duration
        df = pd.DataFrame([valid_row])
        errors = validate_input_schema(df, feature_set="set_b")
        assert errors == []  # should NOT fail — duration is silently removed

    def test_empty_dataframe_returns_error(self):
        """Empty DataFrame should return a non-empty error list (not raise)."""
        empty = pd.DataFrame(columns=_REQUIRED_INPUT_COLS)
        errors = validate_input_schema(empty, feature_set="set_b")
        assert len(errors) > 0
        assert any("empty" in e.lower() for e in errors)

    def test_batch_with_extra_columns_passes(self, valid_row):
        """Extra unknown columns (other than duration) should not cause failures."""
        valid_row["extra_col"] = "whatever"
        df = pd.DataFrame([valid_row])
        errors = validate_input_schema(df, feature_set="set_b")
        assert errors == []

    def test_batch_csv_missing_required_column_returns_error_list(self, valid_df):
        df_bad = valid_df.drop(columns=["poutcome"])
        errors = validate_input_schema(df_bad, feature_set="set_b")
        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any("poutcome" in e for e in errors)

    def test_five_row_valid_df_returns_empty_errors(self, valid_df):
        errors = validate_input_schema(valid_df, feature_set="set_b")
        assert errors == []
