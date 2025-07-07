"""
Debug script to examine model features and structure
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Examine model structure and feature requirements"""
    # Find the latest model
    models_dir = Path("models/AAPL")
    model_files = list(models_dir.glob("model_*.joblib"))
    if not model_files:
        logger.error("No model files found")
        return
        
    latest_model = max(model_files, key=os.path.getmtime)
    model_timestamp = latest_model.stem.split('_')[-1]
    
    # Load the model
    logger.info(f"Loading model: {latest_model}")
    model = joblib.load(latest_model)
    
    # Initialize feature_columns
    feature_columns = []
    
    # Load metadata
    metadata_file = models_dir / f"metadata_AAPL_{model_timestamp}.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            feature_columns = metadata.get('feature_columns', [])
            logger.info(f"Feature columns from metadata: {feature_columns}")
    else:
        logger.warning(f"Metadata file not found: {metadata_file}")
    
    # Examine the model structure
    logger.info(f"Model type: {type(model)}")
    if hasattr(model, 'steps'):
        logger.info("Pipeline steps:")
        for i, (name, step) in enumerate(model.steps):
            logger.info(f"  {i}. {name}: {type(step)}")
            
    # Try to extract feature names from the model if available
    if hasattr(model, 'feature_names_in_'):
        logger.info(f"Model feature names: {model.feature_names_in_.tolist()}")
    elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_names_in_'):
        logger.info(f"Model feature names: {model.steps[-1][1].feature_names_in_.tolist()}")
    
    # Load a sample data file to compare
    sample_file = Path("data/AAPL/202506/data-20250623.csv")
    if sample_file.exists():
        df = pd.read_csv(sample_file)
        logger.info(f"Sample data columns: {df.columns.tolist()}")
        
        # Check which features are missing
        if feature_columns:
            missing = set(feature_columns) - set(df.columns)
            logger.info(f"Missing features: {missing}")
            
            # Check data types
            if feature_columns:
                logger.info("Feature data types:")
                for col in feature_columns:
                    if col in df.columns:
                        logger.info(f"  {col}: {df[col].dtype}")
    
    # Get model feature names
    model_features = []
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_.tolist()
    elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_names_in_'):
        model_features = model.steps[-1][1].feature_names_in_.tolist()
    
    # Try to create a minimal working example
    if sample_file.exists() and model_features:
        try:
            df = pd.read_csv(sample_file)
            logger.info(f"Missing features in sample data: {set(model_features) - set(df.columns)}")
            
            # Create feature engineering pipeline if needed
            from feature_engineer import FeatureEngineer
            engineer = FeatureEngineer()
            df = engineer.enrich_dataframe(df)
            
            # Add missing columns with zeros
            for col in model_features:
                if col not in df.columns:
                    logger.info(f"Adding missing column: {col}")
                    df[col] = 0.0
            
            # Select only the required features in the correct order
            X = df[model_features].copy()
            
            # Handle missing values
            X = X.fillna(0)
            
            logger.info(f"Created feature matrix with shape: {X.shape}")
            
            # Try to make a prediction
            try:
                logger.info("Attempting prediction with sample data...")
                pred = model.predict(X)
                logger.info(f"Prediction successful! Result shape: {pred.shape}")
                logger.info(f"Unique predictions: {np.unique(pred)}")
                
                # Show sample predictions
                proba = model.predict_proba(X)
                results_df = pd.DataFrame({
                    'datetime': df['datetime'] if 'datetime' in df else range(len(X)),
                    'prediction': pred
                })
                for i, cls in enumerate(model.classes_):
                    results_df[f'prob_{cls}'] = proba[:, i]
                results_df['confidence'] = proba.max(axis=1)
                
                logger.info("Sample predictions:")
                logger.info(results_df.tail().to_string())
                
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            logger.error(f"Failed to create minimal example: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
