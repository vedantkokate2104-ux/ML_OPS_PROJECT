"""Unit tests for model artefacts and pure helpers (no database needed)."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import data_access
import modeling

ROOT = Path(__file__).resolve().parents[1]


def test_model_files_present():
    for p in [modeling.STAGE1_MODEL_PATH, modeling.STAGE2_MODEL_PATH, modeling.FEATURES_PATH,
              modeling.CLASS_MAP_PATH, modeling.LABEL_ENCODER_PATH]:
        assert p.exists(), p


def test_models_predict_on_feature_matrix():
    clf, reg, selected, idx_to_label = modeling.load_models()
    with open(modeling.LABEL_ENCODER_PATH, "rb") as f:
        enc = pickle.load(f)
    n = 40
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, tz=modeling.IST),
        "asset": "NIFTY",
        "returns": np.random.default_rng(0).normal(0, 0.01, n),
        "rolling_volatility": 0.01, "rsi": 50.0, "atr": 100.0, "volume_zscore": 0.0,
        "open": 22000.0, "high": 22100.0, "low": 21900.0, "close": 22050.0,
    })
    df = modeling.add_regime_labels(df)
    df, cols, _ = modeling.build_feature_matrix(df, encoder=enc, fit_encoder=False)
    out = modeling.predict(df, cols)
    assert set(out["predicted_direction"].unique()) <= {-1, 0, 1}
    assert out["confidence_score"].between(0, 1).all()
    assert (out["predicted_magnitude"] >= 0).all()


def test_asset_lists_consistent():
    assert data_access.KNOWN_ASSETS == modeling.KNOWN_ASSETS


def test_get_client_requires_uri(monkeypatch):
    monkeypatch.delenv("MONGO_URI", raising=False)
    try:
        data_access.get_client()
    except ValueError:
        return
    raise AssertionError("expected ValueError without MONGO_URI")
