"""
mlflow_tracking.py
EventOracle – MLflow experiment tracking & model registry

Logs the trained two-stage LightGBM models to MLflow together with their
hyper-parameters, SHAP-selected features, feature importances, and an
out-of-pipeline evaluation of the stored predictions against realised
market outcomes (labels built with modeling.label_events, P&L with
modeling.backtest – the same definitions used in training).

Usage:
    # MLflow server from docker-compose (http://localhost:5000) or any URI
    export MLFLOW_TRACKING_URI=http://localhost:5000
    python mlflow_tracking.py                 # log + register current models
    python mlflow_tracking.py --no-register   # skip model registry

modeling.py --train also calls log_training_run() automatically when
MLFLOW_TRACKING_URI is set, so every retrain is tracked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import mlflow  # noqa: E402

import data_access as da  # noqa: E402
import modeling  # noqa: E402

log = logging.getLogger("eventoracle.mlflow")

EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "EventOracle-LightGBM")
STAGE1_NAME = "eventoracle-stage1-impact-classifier"
STAGE2_NAME = "eventoracle-stage2-magnitude-regressor"


def _setup() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)


def _model_version_tag() -> str:
    h = hashlib.sha1()
    for p in [modeling.STAGE1_MODEL_PATH, modeling.STAGE2_MODEL_PATH, modeling.FEATURES_PATH]:
        h.update(p.read_bytes())
    return f"lgbm-{h.hexdigest()[:10]}"


def _log_params(clf, reg, selected: list[str]) -> None:
    mlflow.log_params({f"stage1_{k}": v for k, v in clf.get_params().items() if v is not None})
    mlflow.log_params({f"stage2_{k}": v for k, v in reg.get_params().items() if v is not None})
    mlflow.log_params({
        "n_selected_features": len(selected),
        "neutral_threshold": modeling.NEUTRAL_THRESHOLD,
        "txn_cost": modeling.TXN_COST,
        "n_regimes": modeling.N_REGIMES,
        "cv_splits": modeling.N_SPLITS_CV,
        "shap_drop_pct": modeling.SHAP_DROP_PCT,
    })


def _log_feature_artifacts(clf, selected: list[str]) -> None:
    imp = (pd.DataFrame({"feature": selected, "importance": clf.feature_importances_})
           .sort_values("importance", ascending=False))
    with tempfile.TemporaryDirectory() as tmp:
        imp.to_csv(Path(tmp) / "feature_importance.csv", index=False)
        (Path(tmp) / "selected_features.json").write_text(json.dumps(selected, indent=2))
        mlflow.log_artifacts(tmp, artifact_path="features")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = imp.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(top["feature"], top["importance"], color="#3b82f6")
        ax.set_title("Stage 1 – top 20 feature importances (split count)")
        fig.tight_layout()
        mlflow.log_figure(fig, "features/feature_importance.png")
        plt.close(fig)
    except ImportError:
        pass


def _log_models(clf, reg, selected: list[str], register: bool) -> None:
    example = pd.DataFrame([np.zeros(len(selected))], columns=selected).astype(float)
    mlflow.lightgbm.log_model(clf, name="stage1_impact_classifier", input_example=example,
                              registered_model_name=STAGE1_NAME if register else None)
    mlflow.lightgbm.log_model(reg, name="stage2_magnitude_regressor", input_example=example,
                              registered_model_name=STAGE2_NAME if register else None)


# ─────────────────────────────────────────────
# EVALUATION ON REALISED OUTCOMES
# ─────────────────────────────────────────────

def evaluate_asset(client, asset: str) -> dict | None:
    """Score stored predictions for one asset against labels built from prices."""
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

    prices = da.latest_rows(client, "prices", asset, limit=5000)
    preds = da.latest_rows(client, "predictions", asset, limit=5000)
    if prices.empty or preds.empty:
        return None

    labels = modeling.label_events(prices[["timestamp", "asset", "close", "returns"]])
    labels["date"] = labels["timestamp"].dt.date
    preds["date"] = preds["timestamp"].dt.date
    df = preds[["date", "timestamp", "asset", "predicted_direction", "predicted_magnitude", "confidence_score"]].merge(
        labels[["date", "impact_label", "magnitude_label", "forward_return_t1"]], on="date", how="inner")
    df = df.dropna(subset=["predicted_direction", "impact_label"])
    if len(df) < 10:
        return None

    y, yhat = df["impact_label"].astype(int), df["predicted_direction"].astype(int)
    bt = modeling.backtest(df.copy())
    sr = bt["strategy_return"].dropna()
    majority = y.value_counts(normalize=True).iloc[0]
    return {
        "n_evaluated": len(df),
        "accuracy": accuracy_score(y, yhat),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "baseline_majority_accuracy": majority,
        "magnitude_mae": mean_absolute_error(df["magnitude_label"], df["predicted_magnitude"].fillna(0)),
        "mean_confidence": df["confidence_score"].mean(),
        "backtest_total_return": sr.sum(),
        "backtest_sharpe": (sr.mean() / (sr.std() + 1e-9)) * np.sqrt(252) if len(sr) > 1 else 0.0,
        "backtest_win_rate": (sr > 0).mean() if len(sr) else 0.0,
    }


def log_current_models(register: bool = True) -> str:
    _setup()
    clf, reg, selected, idx_to_label = modeling.load_models()
    with open(modeling.LABEL_ENCODER_PATH, "rb") as f:
        enc = pickle.load(f)
    client = da.get_client()
    version = _model_version_tag()

    with mlflow.start_run(run_name=f"evaluate-{version}") as run:
        mlflow.set_tags({
            "model_version": version, "model_type": "two-stage LightGBM",
            "stage": "evaluation", "data_source": "MongoDB Atlas (eventoracle)",
            "evaluation_note": "Stored predictions vs realised 2-session outcomes; "
                               "may include the in-sample training period.",
        })
        _log_params(clf, reg, selected)
        mlflow.log_dict({str(k): int(v) for k, v in idx_to_label.items()}, "class_map.json")
        mlflow.log_dict({"assets": list(map(str, enc.classes_))}, "asset_encoder.json")
        _log_feature_artifacts(clf, selected)

        rows = []
        for asset in da.KNOWN_ASSETS:
            m = evaluate_asset(client, asset)
            if not m:
                log.warning(f"{asset}: not enough data to evaluate")
                continue
            rows.append({"asset": asset, **m})
            with mlflow.start_run(run_name=asset, nested=True):
                mlflow.set_tags({"asset": asset, "model_version": version})
                mlflow.log_metrics(m)
            log.info(f"{asset:<11} acc={m['accuracy']:.3f}  f1={m['macro_f1']:.3f}  sharpe={m['backtest_sharpe']:+.2f}")

        if rows:
            table = pd.DataFrame(rows)
            weights = table["n_evaluated"]
            mlflow.log_metrics({
                "portfolio_accuracy": float(np.average(table["accuracy"], weights=weights)),
                "portfolio_macro_f1": float(np.average(table["macro_f1"], weights=weights)),
                "portfolio_baseline_accuracy": float(np.average(table["baseline_majority_accuracy"], weights=weights)),
                "portfolio_mean_sharpe": float(table["backtest_sharpe"].mean()),
                "assets_evaluated": len(table),
            })
            mlflow.log_table(table.round(6), "evaluation/per_asset_metrics.json")

        _log_models(clf, reg, selected, register)
        return run.info.run_id


def log_training_run(clf, reg, selected: list[str], metrics: dict) -> None:
    """Called from modeling.train() when MLFLOW_TRACKING_URI is set."""
    _setup()
    with mlflow.start_run(run_name="train") as run:
        mlflow.set_tags({"stage": "training", "model_type": "two-stage LightGBM"})
        _log_params(clf, reg, selected)
        mlflow.log_metrics(metrics)
        _log_feature_artifacts(clf, selected)
        _log_models(clf, reg, selected, register=True)
        log.info(f"MLflow training run logged: {run.info.run_id}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # MLflow prints emoji run links (Windows consoles)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Log EventOracle models & evaluation to MLflow")
    parser.add_argument("--no-register", action="store_true", help="skip the MLflow model registry")
    args = parser.parse_args()
    run_id = log_current_models(register=not args.no_register)
    print(f"Logged MLflow run {run_id} to {mlflow.get_tracking_uri()}")
