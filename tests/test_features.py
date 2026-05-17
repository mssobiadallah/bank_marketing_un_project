"""
test_features.py — Unit tests for src/features.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import add_features, encode_target, get_feature_lists
from src.config import ENGINEERED_FEATURE_NAMES, TARGET_COL


class TestEncodeTarget:
    def test_maps_yes_to_1(self, sample_df):
        df = encode_target(sample_df)
        yes_rows = sample_df[sample_df[TARGET_COL] == "yes"].index
        assert (df.loc[yes_rows, TARGET_COL] == 1).all()

    def test_maps_no_to_0(self, sample_df):
        df = encode_target(sample_df)
        no_rows = sample_df[sample_df[TARGET_COL] == "no"].index
        assert (df.loc[no_rows, TARGET_COL] == 0).all()

    def test_missing_target_raises(self, sample_df):
        df = sample_df.drop(columns=[TARGET_COL])
        with pytest.raises(KeyError):
            encode_target(df)

    def test_does_not_mutate_input(self, sample_df):
        original_val = sample_df[TARGET_COL].iloc[0]
        encode_target(sample_df)
        assert sample_df[TARGET_COL].iloc[0] == original_val


class TestAddFeatures:
    def test_adds_exactly_9_new_columns(self, sample_df):
        n_before = len(sample_df.columns)
        df = add_features(sample_df)
        assert len(df.columns) == n_before + 9

    def test_all_engineered_columns_present(self, sample_df):
        df = add_features(sample_df)
        for col in ENGINEERED_FEATURE_NAMES:
            assert col in df.columns, f"Missing engineered feature: {col}"

    def test_was_previously_contacted_zero_when_pdays_999(self, sample_df):
        df = add_features(sample_df)
        # All sample_df rows have pdays=999
        assert (df["was_previously_contacted"] == 0).all()

    def test_was_previously_contacted_one_when_pdays_less_than_999(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "pdays"] = 5
        result = add_features(df)
        assert result.loc[0, "was_previously_contacted"] == 1

    def test_previous_contact_success_flag(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "poutcome"] = "success"
        result = add_features(df)
        assert result.loc[0, "previous_contact_success_flag"] == 1
        assert (result.loc[1:, "previous_contact_success_flag"] == 0).all()

    def test_no_duplicate_columns(self, sample_df):
        df = add_features(sample_df)
        assert len(df.columns) == len(set(df.columns))

    def test_does_not_mutate_input(self, sample_df):
        cols_before = list(sample_df.columns)
        add_features(sample_df)
        assert list(sample_df.columns) == cols_before


class TestGetFeatureLists:
    def test_returns_required_keys(self, sample_df):
        df = add_features(sample_df)
        result = get_feature_lists(df, target=TARGET_COL, exclude_duration=False)
        assert "numeric" in result
        assert "categorical" in result
        assert "target" in result

    def test_target_not_in_numeric_or_categorical(self, sample_df):
        df = add_features(sample_df)
        result = get_feature_lists(df, target=TARGET_COL, exclude_duration=False)
        assert TARGET_COL not in result["numeric"]
        assert TARGET_COL not in result["categorical"]

    def test_exclude_duration_removes_duration_from_numeric(self, sample_df):
        df = add_features(sample_df)
        result_b = get_feature_lists(df, target=TARGET_COL, exclude_duration=True)
        assert "duration" not in result_b["numeric"]

    def test_include_duration_keeps_duration_in_numeric(self, sample_df):
        df = add_features(sample_df)
        result_a = get_feature_lists(df, target=TARGET_COL, exclude_duration=False)
        assert "duration" in result_a["numeric"]

    def test_no_column_in_both_lists(self, sample_df):
        df = add_features(sample_df)
        result = get_feature_lists(df, target=TARGET_COL, exclude_duration=True)
        overlap = set(result["numeric"]) & set(result["categorical"])
        assert len(overlap) == 0, f"Columns in both numeric and categorical: {overlap}"
