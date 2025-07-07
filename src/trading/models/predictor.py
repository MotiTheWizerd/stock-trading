"""trading.models.predictor

StockPredictor – model loading & inference wrapped in a reusable class.

This code was extracted from the previous monolithic `scripts/predict.py` so
that other parts of the application (CLI, back-tester, notebooks) can import
`trading.models.StockPredictor` without depending on the full CLI script.

The implementation is *identical* to the original; only top-level logging
setup has been removed so that applications can configure logging themselves.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

__all__ = ["StockPredictor"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities (kept private – not exported)
# ---------------------------------------------------------------------------

def _norm(col: str) -> str:
    """Normalise a column name for loose matching.

    • lower-case
    • strip whitespace
    • remove underscores / hyphens
    """
    return (
        str(col)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

# ---------------------------------------------------------------------------
# Core predictor class
# ---------------------------------------------------------------------------

class StockPredictor:
    """Load a trained model and make predictions on new feature data."""

    def __init__(self, ticker: str, model_dir: str | None = None):
        # Allow passing a direct model file path instead of a ticker
        ticker_str = str(ticker)
        if (ticker_str.endswith(".joblib") or ticker_str.endswith(".pkl")) and Path(ticker_str).exists():
            model_path = Path(ticker_str)
            self.model_dir = model_path.parent
            self.ticker = model_path.stem.split("_")[0].upper()
        else:
            self.ticker = ticker_str.upper()
            self.model_dir = Path(model_dir) if model_dir else Path("models") / self.ticker
            self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None  # will be populated by _load_model()
        self.model_classes: list[str] = []
        self.feature_columns: list[str] = []
        self.metadata: dict[str, object] = {}

        self._load_model()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _load_model(self) -> None:
        """Locate and load the newest *.joblib / *.pkl model."""
        # Resolve file vs. directory cases first
        if self.model_dir.is_file() and self.model_dir.suffix in {".joblib", ".pkl"}:
            model_file = self.model_dir
            self.model_dir = model_file.parent
        else:
            if not self.model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

            logger.info("Searching for model files in: %s", self.model_dir.absolute())
            all_files = list(self.model_dir.glob("*.joblib")) + list(self.model_dir.glob("*.pkl"))
            logger.info("Found %d model files: %s", len(all_files), [f.name for f in all_files])
            if not all_files:
                raise FileNotFoundError(f"No model files found in {self.model_dir.absolute()}")

            # Prefer base models (exclude '_calibrated.joblib') if present
            base_models = [f for f in all_files if not f.name.endswith("_calibrated.joblib")]
            search_pool = base_models or all_files
            model_file = max(search_pool, key=lambda f: f.stat().st_mtime)

        logger.info("✅ Loading model from: %s", model_file)
        self.model = joblib.load(model_file)
        self.model_classes = self.model.classes_.tolist()
        logger.info("Model classes: %s", self.model_classes)

        # Optional metadata (JSON side-car)
        metadata_file = model_file.with_suffix(".json")
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as fh:
                self.metadata = json.load(fh)
            if "feature_columns" in self.metadata:
                self.feature_columns = self.metadata["feature_columns"]
                logger.info("Extracted %d feature columns from metadata", len(self.feature_columns))

        # Fallback – introspect the model for feature names
        if not self.feature_columns:
            logger.info("No feature columns in metadata; attempting extraction from model…")
            if hasattr(self.model, "feature_names_in_"):
                self.feature_columns = list(self.model.feature_names_in_)
            elif (
                hasattr(self.model, "named_steps")
                and self.model.named_steps.get("model") is not None
                and hasattr(self.model.named_steps["model"], "feature_names_in_")
            ):
                self.feature_columns = list(self.model.named_steps["model"].feature_names_in_)
            else:
                raise ValueError(
                    "Model does not contain feature names; re-train with feature_columns in metadata."
                )
            logger.info("Recovered %d feature columns from model", len(self.feature_columns))

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the input `df` enriched with predictions, probabilities & risk."""
        df_copy = df.copy()

        logger.info("Original columns: %s", df_copy.columns.tolist())
        logger.info("Model expects features: %s", self.feature_columns)

        # Soft normalisation (replace _ with space) for quick look-ups
        df_copy.columns = [str(col).replace("_", " ").strip() for col in df_copy.columns]

        # Ensure datetime column has correct dtype
        if "datetime" in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy["datetime"]):
            try:
                df_copy["datetime"] = pd.to_datetime(df_copy["datetime"])
                logger.info("Converted datetime column to datetime64[ns]")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to convert datetime column: %s", exc)

        # -----------------------------------------------------------------
        # 1. Attempt direct column alignment
        # -----------------------------------------------------------------
        normalized_feature_columns = [col.replace("_", " ").strip() for col in self.feature_columns]
        missing_features = [c for c in normalized_feature_columns if c not in df_copy.columns]

        # -----------------------------------------------------------------
        # 2. Column name mapping heuristics if initial match fails
        # -----------------------------------------------------------------
        if missing_features:
            logger.warning("Missing features: %s", missing_features)
            column_mapping: dict[str, str] = {}
            for col in missing_features:
                variations = [
                    col.lower(),
                    col.upper(),
                    col.replace(" ", ""),
                    col.replace(" ", "_"),
                    col.replace("_", " ").strip(),
                    col.lower().replace(" ", "_"),
                    col.upper().replace(" ", "_"),
                ]
                for var in variations:
                    matches = [c for c in df_copy.columns if c.lower() == var.lower()]
                    if matches:
                        column_mapping[col] = matches[0]
                        logger.info("Mapped '%s' ➞ '%s'", col, matches[0])
                        break
            if column_mapping:
                df_copy = df_copy.rename(columns={v: k for k, v in column_mapping.items()})
                missing_features = [c for c in normalized_feature_columns if c not in df_copy.columns]

        # -----------------------------------------------------------------
        # 3. Last resort – add zero-filled columns
        # -----------------------------------------------------------------
        if missing_features:
            logger.warning("Still missing features after mapping: %s", missing_features)
            for col in missing_features:
                df_copy[col] = 0.0

        # -----------------------------------------------------------------
        # 4. Normalise names strictly to map to model order
        # -----------------------------------------------------------------
        norm_to_raw = {_norm(c): c for c in df_copy.columns}
        ordered_pairs: list[tuple[str, str]] = [
            (model_col, norm_to_raw[_norm(model_col)])
            for model_col in self.feature_columns
            if _norm(model_col) in norm_to_raw
        ]
        if not ordered_pairs:
            raise ValueError("No matching features between model and data after normalisation")

        X = df_copy[[raw for _mc, raw in ordered_pairs]].copy()
        X.columns = [mc for mc, _raw in ordered_pairs]  # rename to model’s original names

        logger.info("Feature matrix shape: %s", X.shape)

        # Fill any remaining NaNs
        if X.isna().any().any():
            X = X.fillna(0)

        # -----------------------------------------------------------------
        # 5. Inference
        # -----------------------------------------------------------------
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        confidences = np.max(probabilities, axis=1)
        
        # Extract key technical indicators for tie-breaking if they exist
        rsi_values = X["RSI_14"].values if "RSI_14" in X.columns else None
        bollinger_upper = X["Bollinger_Upper"].values if "Bollinger_Upper" in X.columns else None
        bollinger_lower = X["Bollinger_Lower"].values if "Bollinger_Lower" in X.columns else None
        bollinger_mid = X["Bollinger_Mid"].values if "Bollinger_Mid" in X.columns else None
        macd = X["MACD"].values if "MACD" in X.columns else None
        macd_signal = X["MACD_Signal"].values if "MACD_Signal" in X.columns else None
        
        # Smart prediction extraction with tie-breaking
        smart_predictions = []
        prob_diffs = []
        
        for i, probs in enumerate(probabilities):
            # Get class indices sorted by probability (highest first)
            sorted_indices = np.argsort(probs)[::-1]
            
            # Get top two probabilities and their classes
            top_class_idx = sorted_indices[0]
            second_class_idx = sorted_indices[1]
            top_prob = probs[top_class_idx]
            second_prob = probs[second_class_idx]
            prob_diff = top_prob - second_prob
            prob_diffs.append(prob_diff)
            
            # If probabilities are very close (within 0.1), use indicators for tie-breaking
            if prob_diff < 0.1:
                top_class = self.model_classes[top_class_idx]
                second_class = self.model_classes[second_class_idx]
                
                # Apply tie-breaking logic using technical indicators
                if rsi_values is not None:
                    rsi = rsi_values[i]
                    # RSI-based tie-breaking
                    if rsi < 30 and second_class == "BUY":  # Oversold condition
                        smart_predictions.append("BUY")
                        continue
                    elif rsi > 70 and second_class == "SELL":  # Overbought condition
                        smart_predictions.append("SELL")
                        continue
                
                # Bollinger Bands tie-breaking
                if bollinger_upper is not None and bollinger_lower is not None and "Close" in df.columns:
                    close = df["Close"].iloc[i]
                    upper = bollinger_upper[i]
                    lower = bollinger_lower[i]
                    
                    # Price near upper band suggests SELL
                    if close > upper * 0.98 and second_class == "SELL":
                        smart_predictions.append("SELL")
                        continue
                    # Price near lower band suggests BUY
                    elif close < lower * 1.02 and second_class == "BUY":
                        smart_predictions.append("BUY")
                        continue
                
                # MACD crossover tie-breaking
                if macd is not None and macd_signal is not None:
                    macd_val = macd[i]
                    signal = macd_signal[i]
                    
                    # MACD above signal line suggests BUY
                    if macd_val > signal and second_class == "BUY":
                        smart_predictions.append("BUY")
                        continue
                    # MACD below signal line suggests SELL
                    elif macd_val < signal and second_class == "SELL":
                        smart_predictions.append("SELL")
                        continue
            
            # If no tie-breaking applied, use the highest probability class
            smart_predictions.append(self.model_classes[top_class_idx])

        # -----------------------------------------------------------------
        # 6. Assemble result frame (preserve original datetime & price columns)
        # -----------------------------------------------------------------
        essential_cols: list[str] = []
        if "datetime" in df.columns:
            essential_cols.append("datetime")
        for price_col in ["Close", "Open", "High", "Low", "Adj Close", "Price"]:
            if price_col in df.columns:
                essential_cols.append(price_col)
        result_df = df[essential_cols].copy() if essential_cols else pd.DataFrame(index=df.index)
        
        # Ensure Price column is always available (copy from Close if needed)
        if "Close" in df.columns and "Price" not in result_df.columns:
            result_df["Price"] = df["Close"]
            logger.info("Created 'Price' column from 'Close' values")
        
        # Check if Price column has valid values
        if "Price" in result_df.columns and (result_df["Price"].isna().all() or (result_df["Price"] == 0).all()):
            logger.warning("Price column contains all NaN or zero values!")

        # Add prediction and confidence columns
        result_df["prediction"] = smart_predictions  # Use smart predictions with tie-breaking
        result_df["confidence"] = confidences
        result_df["prob_diff"] = prob_diffs  # Add probability difference for transparency
        
        # Add probability columns for each class
        for idx, class_name in enumerate(self.model.classes_):
            result_df[f"prob_{class_name}"] = probabilities[:, idx]
        
        # Calculate risk score based on confidence and prediction type
        # Higher risk for lower confidence and non-NONE predictions
        result_df["risk_score"] = 1 - result_df["confidence"]
        
        # Adjust risk tiers: Low Risk (LR) <= 0.4, High Risk (HR) > 0.4
        result_df["risk_tier"] = np.where(result_df["risk_score"] <= 0.4, "LR", "HR")
        result_df["risk_emoji"] = np.where(result_df["risk_tier"] == "LR", "🟢", "🔴")
        
        # Create prediction_with_risk that respects high confidence predictions
        result_df["prediction_with_risk"] = result_df.apply(
            lambda r: f"{r['prediction']} {r['risk_emoji']} {r['risk_tier']}",
            axis=1,
        )

        # Quick distribution logs (INFO level)
        pred_counts = pd.Series(predictions).value_counts().to_dict()
        logger.info("Prediction distribution: %s", pred_counts)
        risk_counts = result_df["risk_tier"].value_counts().to_dict()
        logger.info("Risk tier distribution: %s", risk_counts)
        
        # Log price column statistics
        price_cols = [col for col in result_df.columns if col in ["Price", "Close", "Open", "High", "Low", "Adj Close"]]
        if price_cols:
            logger.info("Price columns available: %s", price_cols)
            for col in price_cols:
                if result_df[col].isna().all() or (result_df[col] == 0).all():
                    logger.warning("%s column contains all NaN or zero values!", col)
        
        # Add key technical indicators directly (not prefixed)
        key_indicators = {
            "RSI": "RSI_14", 
            "Bollinger_Upper": "Bollinger_Upper", 
            "Bollinger_Lower": "Bollinger_Lower", 
            "Bollinger_Mid": "Bollinger_Mid",
            "MACD": "MACD",
            "MACD_Signal": "MACD_Signal",
            "EMA5": "EMA_5",
            "EMA10": "EMA_10"
        }
        
        for display_name, col_name in key_indicators.items():
            if col_name in X.columns:
                result_df[display_name] = X[col_name].values
                
        # Add prediction confidence distribution
        result_df["confidence_level"] = pd.cut(
            result_df["confidence"],
            bins=[0, 0.5, 0.8, 0.95, 1.0],
            labels=["Very Low", "Low", "Medium", "High"]
        )

        return result_df
