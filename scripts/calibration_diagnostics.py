import argparse
import logging
from pathlib import Path
from datetime import datetime

import joblib
import sys

# Ensure project src directory is on PYTHONPATH for local execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'src'))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from trading.features.feature_engineer import FeatureEngineer
from trading.io.paths import get_data_path, get_models_path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_latest_model(ticker: str):
    models_path = get_models_path(ticker)
    model_files = sorted(models_path.glob(f"model_{ticker}_*.joblib"), key=lambda p: p.stat().st_mtime)
    if not model_files:
        raise FileNotFoundError(f"No model files found for {ticker}")
    return model_files[-1]


def load_validation_set(ticker: str, test_size: float = 0.2, target_col: str = 'signal'): 
    """Load recent data for calibration diagnostics.

    Uses the same feature pipeline as training and returns X_val, y_val.
    """
    data_path = get_data_path(ticker)
    date_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not date_dirs:
        raise FileNotFoundError("No data directories found for validation.")

    # Load last 30 days (or all if fewer)
    data_files, signals_files = [], []
    for d in date_dirs[-30:]:
        df_path, sig_path = d / "data.csv", d / "signals.csv"
        if df_path.exists() and sig_path.exists():
            data_files.append(df_path)
            signals_files.append(sig_path)
    if not data_files:
        raise FileNotFoundError("No paired data.csv and signals.csv files for validation.")

    df = pd.concat([pd.read_csv(f, parse_dates=["datetime"]) for f in data_files])
    sig_dfs = []
    for f in signals_files:
        s = pd.read_csv(f, parse_dates=["datetime"])
        if target_col not in s.columns:
            logger.warning("Target column '%s' not found in %s – skipping", target_col, f.name)
            continue
        sig_dfs.append(s[["datetime", target_col]])
    if not sig_dfs:
        raise KeyError(f"No signal files contained the target column '{target_col}'.")
    sig = pd.concat(sig_dfs)
    merged = pd.merge(df, sig, on="datetime", how="inner")

    # Basic sanity filter: drop rows without target labels
    merged = merged.dropna(subset=[target_col])

    # Map textual labels to numeric (BUY=1, SELL=0); drop other rows BEFORE feature engineering
    label_map = {'BUY': 1, 'SELL': 0}
    merged = merged[merged[target_col].isin(label_map)].copy()
    merged[target_col] = merged[target_col].map(label_map).astype(int)

    # Now run feature engineering so X aligns perfectly with y
    fe = FeatureEngineer()
    X_full = fe.enrich_dataframe(merged)
    y_full = merged[target_col].values



    # Simple temporal split: last test_size proportion as validation
    split_idx = int(len(X_full) * (1 - test_size))
    X_train, X_val = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    y_train, y_val = y_full[:split_idx], y_full[split_idx:]

    return X_train, y_train, X_val, y_val


def fit_calibrator(method: str, y_proba_train, y_train):
    if method == "platt":
        # Platt scaling via logistic regression on probabilities
        lr = LogisticRegression(max_iter=1000)
        lr.fit(y_proba_train.reshape(-1, 1), y_train)
        return lr
    elif method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_proba_train, y_train)
        return iso
    else:
        raise ValueError("Unknown calibration method: " + method)


def apply_calibrator(calibrator, y_proba):
    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
    elif isinstance(calibrator, IsotonicRegression):
        return calibrator.transform(y_proba)
    else:
        raise ValueError("Unsupported calibrator type")


def reliability_plot(y_true, y_prob, output_path: Path, title: str):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=20)
    plt.figure(figsize=(4, 4))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Empirical Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Calibration diagnostics")
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument("--method", choices=["platt", "isotonic"], default="platt", help="Calibration method")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation size fraction")
    parser.add_argument("--target-col", default="signal", help="Name of target column in signals.csv")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    model_path = load_latest_model(ticker)
    model = joblib.load(model_path)
    logger.info("Loaded model: %s", model_path.name)

    # Load data
    X_train, y_train, X_val, y_val = load_validation_set(ticker, args.test_size, args.target_col)

    # Ensure feature alignment with the trained model
    if hasattr(model, 'feature_names_in_'):
        known = model.feature_names_in_
        X_train = X_train[known]
        X_val = X_val[known]
    logger.info("Validation samples: %d", len(X_val))

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not support predict_proba()")

    y_proba_val_raw = model.predict_proba(X_val)[:, 1]
    y_proba_train_raw = model.predict_proba(X_train)[:, 1]

    # Fit calibrator on held-out train part (could also be cross-val)
    calibrator = fit_calibrator(args.method, y_proba_train_raw, y_train)

    y_proba_val_cal = apply_calibrator(calibrator, y_proba_val_raw)

    # Metrics
    metrics = {
        "brier_raw": brier_score_loss(y_val, y_proba_val_raw),
        "brier_cal": brier_score_loss(y_val, y_proba_val_cal),
        "logloss_raw": log_loss(y_val, y_proba_val_raw, labels=[0, 1]),
        "logloss_cal": log_loss(y_val, y_proba_val_cal, labels=[0, 1]),
    }
    logger.info("Brier raw=%.4f  cal=%.4f", metrics["brier_raw"], metrics["brier_cal"])
    logger.info("LogLoss raw=%.4f  cal=%.4f", metrics["logloss_raw"], metrics["logloss_cal"])

    # Plots dir
    plots_dir = get_models_path(ticker) / "calibration_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reliability_plot(y_val, y_proba_val_raw, plots_dir / f"reliability_raw_{ts}.png", f"Reliability Raw {ticker}")
    reliability_plot(y_val, y_proba_val_cal, plots_dir / f"reliability_cal_{ts}.png", f"Reliability Calibrated {ticker}")

    # Save calibrator artefact next to model
    calib_path = model_path.with_name(model_path.stem + "_calibrated.joblib")
    joblib.dump(calibrator, calib_path)
    logger.info("Saved calibrator -> %s", calib_path.name)

    # Log summary
    logger.info("Diagnostics complete. Plots saved in %s", plots_dir)


if __name__ == "__main__":
    main()
