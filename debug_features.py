"""
Debug script to identify the exact mismatch between model features and data columns.
"""

import joblib
import pandas as pd
import json
from pathlib import Path

def debug_features():
    # Load the model
    model_path = Path("models/AAPL/model_AAPL_20250707_142508.joblib")
    metadata_path = Path("models/AAPL/metadata_AAPL_20250707_142508.json")
    data_path = Path("data/AAPL/latest/data.csv")
    
    print("=== MODEL INVESTIGATION ===")
    
    # Load model
    model = joblib.load(model_path)
    print(f"Model type: {type(model)}")
    
    # Check if model has feature names
    if hasattr(model, 'feature_names_in_'):
        print(f"Model feature_names_in_: {model.feature_names_in_}")
    else:
        print("Model does not have feature_names_in_ attribute")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    expected_features = metadata['feature_columns']
    print(f"\nExpected features from metadata ({len(expected_features)}):")
    for i, feature in enumerate(expected_features):
        print(f"  {i+1:2d}. '{feature}' (type: {type(feature).__name__})")
    
    print("\n=== DATA INVESTIGATION ===")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    print(f"\nActual columns in data ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. '{col}' (type: {type(col).__name__})")
    
    print("\n=== COMPARISON ===")
    
    # Compare columns
    missing_in_data = []
    for feature in expected_features:
        if feature not in df.columns:
            missing_in_data.append(feature)
    
    extra_in_data = []
    for col in df.columns:
        if col not in expected_features:
            extra_in_data.append(col)
    
    print(f"\nMissing in data ({len(missing_in_data)}):")
    for feature in missing_in_data:
        print(f"  - '{feature}'")
    
    print(f"\nExtra in data ({len(extra_in_data)}):")
    for col in extra_in_data:
        print(f"  + '{col}'")
    
    # Check for case-insensitive matches
    print("\n=== CASE-INSENSITIVE MATCHING ===")
    data_columns_lower = [col.lower() for col in df.columns]
    expected_features_lower = [feature.lower() for feature in expected_features]
    
    for feature in expected_features:
        if feature not in df.columns:
            # Look for case-insensitive match
            feature_lower = feature.lower()
            if feature_lower in data_columns_lower:
                actual_col = df.columns[data_columns_lower.index(feature_lower)]
                print(f"  '{feature}' -> '{actual_col}' (case mismatch)")
            else:
                print(f"  '{feature}' -> NO MATCH")
    
    # Check for exact string representation
    print("\n=== STRING ANALYSIS ===")
    print("Expected features (with repr):")
    for feature in expected_features[:5]:  # Show first 5
        print(f"  '{feature}' -> {repr(feature)}")
    
    print("\nData columns (with repr):")
    for col in df.columns[:5]:  # Show first 5
        print(f"  '{col}' -> {repr(col)}")

if __name__ == "__main__":
    debug_features()
