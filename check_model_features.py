"""
Check model features and data columns.

This script checks the feature columns in the model metadata and compares them with the columns in the data file.
"""

import json
import joblib
import pandas as pd
from pathlib import Path
import sys

def check_model_features(model_dir, data_file):
    """Check model features and data columns."""
    print(f"Checking model features in {model_dir} and data columns in {data_file}")
    
    # Find the latest model file
    model_dir_path = Path(model_dir)
    model_files = list(model_dir_path.glob("*.joblib"))
    if not model_files:
        print(f"No model files found in {model_dir}")
        return
    
    # Sort by modification time (newest first)
    model_file = sorted(model_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
    print(f"Using model file: {model_file}")
    
    # Load the model
    model = joblib.load(model_file)
    
    # Load metadata if available
    metadata_file = model_file.with_suffix(".json")
    feature_columns = []
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Extract feature columns from metadata
        if "feature_columns" in metadata:
            feature_columns = metadata["feature_columns"]
            print(f"\nFeature columns from model metadata ({len(feature_columns)}):")
            print(feature_columns)
    
    # Load the data file
    data_file_path = Path(data_file)
    if not data_file_path.exists():
        print(f"Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file_path)
    print(f"\nData file columns ({len(df.columns)}):")
    print(df.columns.tolist())
    
    # Compare columns
    if feature_columns:
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            print(f"\nMissing columns in data file ({len(missing_columns)}):")
            print(missing_columns)
        else:
            print("\nAll feature columns are present in the data file.")
        
        # Check column types
        print("\nColumn types in data file:")
        for col in feature_columns:
            if col in df.columns:
                print(f"{col}: {df[col].dtype}")
            else:
                print(f"{col}: MISSING")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python check_model_features.py <model_dir> <data_file>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    data_file = sys.argv[2]
    
    check_model_features(model_dir, data_file)
