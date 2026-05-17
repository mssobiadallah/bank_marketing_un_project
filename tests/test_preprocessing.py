"""
test_preprocessing.py — Unit tests for src/preprocessing.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.preprocessing import (
    build_preprocessing_pipeline,
    load_pipeline,
    save_pipeline,
    split_data,
)
from src.config import RANDOM_SEED, TARGET_COL


class TestBuildPreprocessingPipeline:
    def test_returns_column_transformer(self):
        ct = build_preprocessing_pipeline(
            numeric_cols=["age", "campaign"],
            categorical_cols=["job", "marital"],
        )
        assert isinstance(ct, ColumnTransformer)

    def test_fit_transform_produces_array(self, sample_df):
        from src.features import add_features
        df = add_features(sample_df)
        ct = build_preprocessing_pipeline(
            numeric_cols=["age", "campaign"],
            categorical_cols=["job", "marital"],
        )
        X = ct.fit_transform(df)
        assert X.shape[0] == len(df)

    def test_output_shape_increases_with_ohe(self, sample_df):
        """After OHE, number of columns should be >= number of input columns."""
        from src.features import add_features
        df = add_features(sample_df)
        ct = build_preprocessing_pipeline(
            numeric_cols=["age", "campaign"],
            categorical_cols=["job"],
        )
        X = ct.fit_transform(df)
        # 2 numeric + however many OHE columns for 'job'
        assert X.shape[1] >= 3

    def test_handles_unknown_category(self, sample_df):
        """OHE with handle_unknown='ignore' must not raise on unseen category."""
        from src.features import add_features
        df_train = add_features(sample_df)
        ct = build_preprocessing_pipeline(
            numeric_cols=["age"],
            categorical_cols=["job"],
        )
        ct.fit(df_train)

        df_test = df_train.copy()
        df_test.loc[0, "job"] = "alien_job_not_in_train"
        # Should not raise
        X = ct.transform(df_test)
        assert X.shape[0] == len(df_test)


class TestSplitData:
    def test_returns_four_objects(self, processed_df):
        result = split_data(processed_df, target=TARGET_COL, test_size=0.2)
        assert len(result) == 4

    def test_stratification_preserves_target_ratio(self, processed_df):
        X_train, X_test, y_train, y_test = split_data(
            processed_df, target=TARGET_COL, test_size=0.2, random_state=RANDOM_SEED
        )
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.02, (
            f"Target rate mismatch: train={train_rate:.4f}, test={test_rate:.4f}"
        )

    def test_train_plus_test_equals_total(self, processed_df):
        X_train, X_test, y_train, y_test = split_data(processed_df)
        assert len(X_train) + len(X_test) == len(processed_df)


class TestPipelinePersistence:
    def test_save_load_round_trip(self, sample_df, tmp_path):
        from src.features import add_features
        df = add_features(sample_df)
        ct = build_preprocessing_pipeline(
            numeric_cols=["age", "campaign"],
            categorical_cols=["job"],
        )
        ct.fit(df)
        path = tmp_path / "test_pipeline.joblib"
        save_pipeline(ct, path)
        loaded = load_pipeline(path)
        # After reload, transform must produce the same output
        out1 = ct.transform(df)
        out2 = loaded.transform(df)
        assert np.allclose(out1, out2)

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline(tmp_path / "missing.joblib")
