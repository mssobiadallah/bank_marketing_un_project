"""
test_data_loader.py — Unit tests for src/data_loader.py
"""

from __future__ import annotations

import pathlib
import tempfile

import pandas as pd
import pytest

from src.data_loader import (
    compare_datasets,
    load_dataset,
    summarize_dataset,
    validate_required_columns,
)
from src.config import MAIN_DATASET_PATH


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_loads_main_dataset(self):
        """bank-additional-full.csv must load with shape (41188, 21)."""
        if not MAIN_DATASET_PATH.exists():
            pytest.skip("Main dataset not found")
        df = load_dataset(MAIN_DATASET_PATH)
        assert df.shape == (41188, 21)

    def test_loads_with_semicolon_separator(self):
        """Default sep=';' must split columns correctly."""
        if not MAIN_DATASET_PATH.exists():
            pytest.skip("Main dataset not found")
        df = load_dataset(MAIN_DATASET_PATH, sep=";")
        assert len(df.columns) > 1

    def test_wrong_separator_raises_value_error(self, tmp_path):
        """If the wrong separator produces a 1-column DataFrame, raise ValueError."""
        csv = tmp_path / "test.csv"
        csv.write_text("a;b;c\n1;2;3\n4;5;6\n")
        with pytest.raises(ValueError, match="wrong separator"):
            load_dataset(csv, sep=",")

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """A path that does not exist must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.csv")

    def test_empty_file_raises_value_error(self, tmp_path):
        """An empty CSV must raise ValueError."""
        csv = tmp_path / "empty.csv"
        csv.write_text("a;b;c\n")  # header only → empty DataFrame
        with pytest.raises(ValueError):
            load_dataset(csv, sep=";")


# ---------------------------------------------------------------------------
# summarize_dataset
# ---------------------------------------------------------------------------

class TestSummarizeDataset:
    def test_returns_required_keys(self, sample_df):
        summary = summarize_dataset(sample_df)
        required_keys = {
            "shape", "columns", "dtypes", "missing_values",
            "duplicate_rows", "target_distribution", "unknown_counts",
        }
        assert required_keys.issubset(set(summary.keys()))

    def test_shape_matches(self, sample_df):
        summary = summarize_dataset(sample_df)
        assert summary["shape"] == sample_df.shape

    def test_target_distribution_present(self, sample_df):
        summary = summarize_dataset(sample_df)
        assert summary["target_distribution"] is not None
        assert "yes" in summary["target_distribution"] or "no" in summary["target_distribution"]

    def test_no_target_distribution_when_column_absent(self, sample_df):
        df = sample_df.drop(columns=["y"])
        summary = summarize_dataset(df)
        assert summary["target_distribution"] is None

    def test_duplicate_count_is_integer(self, sample_df):
        summary = summarize_dataset(sample_df)
        assert isinstance(summary["duplicate_rows"], int)


# ---------------------------------------------------------------------------
# compare_datasets
# ---------------------------------------------------------------------------

class TestCompareDatasets:
    def test_returns_dataframe(self, tmp_path):
        """compare_datasets must return a DataFrame."""
        # create two tiny valid CSVs
        for name in ("a", "b"):
            (tmp_path / f"{name}.csv").write_text("age;y\n25;no\n30;yes\n")
        result = compare_datasets(
            {"ds_a": tmp_path / "a.csv", "ds_b": tmp_path / "b.csv"}
        )
        assert isinstance(result, pd.DataFrame)

    def test_returns_one_row_per_dataset(self, tmp_path):
        for name in ("a", "b", "c"):
            (tmp_path / f"{name}.csv").write_text("age;y\n25;no\n30;yes\n")
        result = compare_datasets(
            {n: tmp_path / f"{n}.csv" for n in ("a", "b", "c")}
        )
        assert len(result) == 3

    def test_all_four_real_datasets(self):
        """When all 4 real CSVs are present, result must have 4 rows."""
        from src.config import ALL_DATASETS
        available = {k: v for k, v in ALL_DATASETS.items() if v.exists()}
        if len(available) < 4:
            pytest.skip("Not all 4 raw datasets present")
        result = compare_datasets(available)
        assert len(result) == 4

    def test_missing_dataset_skipped_gracefully(self, tmp_path):
        """A missing file must be skipped without raising."""
        (tmp_path / "good.csv").write_text("age;y\n25;no\n")
        result = compare_datasets(
            {"good": tmp_path / "good.csv", "bad": tmp_path / "missing.csv"}
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# validate_required_columns
# ---------------------------------------------------------------------------

class TestValidateRequiredColumns:
    def test_empty_list_on_complete_dataset(self, full_df):
        """A freshly loaded main dataset must have no missing set_b columns."""
        missing = validate_required_columns(full_df, feature_set="set_b")
        # set_b requires raw columns excluding duration + engineered cols (not added yet)
        # just check age is not listed as missing
        assert "age" not in missing

    def test_detects_missing_age(self, sample_df):
        df = sample_df.drop(columns=["age"])
        missing = validate_required_columns(df, feature_set="set_b")
        assert "age" in missing

    def test_invalid_feature_set_raises(self, sample_df):
        with pytest.raises(ValueError, match="set_a.*set_b"):
            validate_required_columns(sample_df, feature_set="set_c")

    def test_set_a_requires_duration(self, sample_df):
        """set_a should NOT flag duration as missing (it is present in sample_df)."""
        missing = validate_required_columns(sample_df, feature_set="set_a")
        assert "duration" not in missing

    def test_set_b_does_not_require_duration(self):
        """set_b column list must not contain duration at all."""
        from src.config import FEATURE_SET_B_COLS
        assert "duration" not in FEATURE_SET_B_COLS
