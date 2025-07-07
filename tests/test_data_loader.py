"""Unit tests for trading.io.data_loader utilities."""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add debug output
print("Python path:", sys.path)

# Import using package path
from trading.io.data_loader import load_latest_data


def _write_sample_csv(tmp_dir: Path, ticker: str, *, price_col: str = "Close") -> Path:
    """Helper to create a sample CSV file for testing."""
    """Create a minimal OHLCV CSV so the loader can pick it up."""
    tdir = tmp_dir / ticker.upper() / "latest"
    tdir.mkdir(parents=True, exist_ok=True)
    sample = pd.DataFrame(
        {
            "datetime": ["2025-07-06 15:30"],
            price_col: [123.45],
            "Open": [120.0],
            "High": [125.0],
            "Low": [119.0],
            "Volume": [1_000_000],
        }
    )
    csv_path = tdir / "data.csv"
    sample.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.parametrize("price_col", ["Close", "Adj Close", "Price"])
def test_load_latest_data_price_column(tmp_path: Path, price_col: str) -> None:
    """Test that load_latest_data correctly handles different price column names."""
    print(f"\nRunning test with price_col='{price_col}'")
    """Loader should always return a numeric Price column regardless of original name."""
    ticker = "TEST"
    _write_sample_csv(tmp_path, ticker, price_col=price_col)

    df = load_latest_data(ticker, data_dir=tmp_path)

    # DataFrame should contain at least one row and a Price column with expected value
    assert not df.empty, "DataLoader returned empty DataFrame"
    assert "Price" in df.columns, "Price column missing after normalisation"

    expected_price = 123.45  # value we seeded
    actual_price = float(df.loc[0, "Price"])
    assert actual_price == pytest.approx(expected_price), "Price value incorrect"
