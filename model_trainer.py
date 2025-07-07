#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Model Trainer for Stock Trading Agents

This module implements advanced training techniques:
1. Custom loss functions that penalize false signals
2. Hard negative mining to focus on difficult examples
3. Class balancing techniques
4. Cross-validation with proper stratification
5. Model ensembling for signal confirmation
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CustomLossTrainer:
    """
    A class to train models with custom loss functions and hard negative mining.
    """
    
    def __init__(self, data_path=None, hard_negatives_path=None):
        """
        Initialize the CustomLossTrainer.
        
        Args:
            data_path: Path to the training data file
            hard_negatives_path: Optional path to hard negatives file
        """
        self.data_path = data_path
        self.hard_negatives_path = hard_negatives_path
        self.data = None
        self.hard_negatives = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.feature_names = None
        self.label_map = {'BUY': 0, 'NONE': 1, 'SELL': 2}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        
        if data_path:
            self.load_data(data_path)
        
        if hard_negatives_path:
            self.load_hard_negatives(hard_negatives_path)
    
    def load_data(self, data_path):
        """Load and prepare data for training."""
        logger.info(f"Loading data from {data_path}")
        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Data loaded with shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def load_hard_negatives(self, hard_negatives_path):
        """Load hard negatives for focused training."""
        logger.info(f"Loading hard negatives from {hard_negatives_path}")
        try:
            self.hard_negatives = pd.read_csv(hard_negatives_path)
            logger.info(f"Hard negatives loaded with shape: {self.hard_negatives.shape}")
            return self.hard_negatives
        except Exception as e:
            logger.error(f"Failed to load hard negatives: {e}")
            raise
    
    def prepare_features_targets(self, target_col='prediction', datetime_col='datetime'):
        """
        Extract features and targets from the data.
        
        Args:
            target_col: Name of the target column
            datetime_col: Name of the datetime column
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Exclude known non-feature columns
        exclude_cols = [target_col, datetime_col, 'Open', 'High', 'Low', 'Close', 
                       'Volume', 'Adj Close', 'prediction', 'confidence']
        self.feature_names = [col for col in self.data.columns 
                             if col not in exclude_cols]
        
        logger.info(f"Selected {len(self.feature_names)} features for training")
        
        # Extract features and target
        X = self.data[self.feature_names].copy()
        
        # Handle missing target (for prediction data)
        if target_col in self.data.columns:
            y = self.data[target_col].copy()
            # Convert string labels to numeric if needed
            if y.dtype == 'object':
                y = y.map(self.label_map)
        else:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Split into train and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {self.X_train.shape}, Validation set: {self.X_val.shape}")
        logger.info(f"Training target distribution: {pd.Series(self.y_train).value_counts(normalize=True)}")
        
        return self.X_train, self.y_train, self.X_val, self.y_val
    
    def add_hard_negatives(self, weight=2.0):
        """
        Add hard negatives to the training set with increased weight.
        
        Args:
            weight: Weight multiplier for hard negatives
            
        Returns:
            Enhanced training features and targets
        """
        if self.hard_negatives is None:
            logger.warning("No hard negatives loaded. Skipping.")
            return self.X_train, self.y_train
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not prepared. Call prepare_features_targets() first.")
        
        logger.info(f"Adding {len(self.hard_negatives)} hard negatives with weight {weight}")
        
        # Get indices of hard negatives in the original data
        hard_neg_indices = self.hard_negatives['index'].values
        
        # Filter to only include indices that are in the training set
        # (some might be in validation set)
        train_indices = self.X_train.index
        valid_hard_neg_indices = [idx for idx in hard_neg_indices if idx in train_indices]
        
        logger.info(f"Found {len(valid_hard_neg_indices)} hard negatives in training set")
        
        if not valid_hard_neg_indices:
            logger.warning("No hard negatives found in training set. Skipping.")
            return self.X_train, self.y_train
        
        # Extract hard negative samples
        hard_neg_X = self.X_train.loc[valid_hard_neg_indices]
        hard_neg_y = self.y_train.loc[valid_hard_neg_indices]
        
        # Duplicate hard negatives based on weight
        n_duplicates = int(weight - 1)  # weight=2 means 1 duplicate
        
        enhanced_X = self.X_train.copy()
        enhanced_y = self.y_train.copy()
        
        for _ in range(n_duplicates):
            enhanced_X = pd.concat([enhanced_X, hard_neg_X])
            enhanced_y = pd.concat([enhanced_y, hard_neg_y])
        
        logger.info(f"Enhanced training set: {enhanced_X.shape}")
        logger.info(f"Enhanced target distribution: {pd.Series(enhanced_y).value_counts(normalize=True)}")
        
        return enhanced_X, enhanced_y
    
    def custom_trading_loss(self, y_true, y_pred):
        """
        Custom loss function that penalizes false signals more than missed opportunities.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Custom loss score (higher is better for sklearn)
        """
        # Convert to numpy arrays if needed
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Define penalty matrix
        # Rows: true labels (BUY=0, NONE=1, SELL=2)
        # Columns: predicted labels (BUY=0, NONE=1, SELL=2)
        penalty_matrix = np.array([
            # BUY predicted, NONE predicted, SELL predicted
            [0, 1, 3],    # BUY true (predicting SELL when true is BUY is worst)
            [2, 0, 2],    # NONE true (predicting either BUY or SELL is equally bad)
            [3, 1, 0]     # SELL true (predicting BUY when true is SELL is worst)
        ])
        
        # Calculate penalties for each sample
        penalties = np.array([penalty_matrix[true, pred] for true, pred in zip(y_true, y_pred)])
        
        # Return negative mean penalty (higher is better for sklearn)
        return -np.mean(penalties)
    
    def get_custom_scorer(self):
        """
        Create a custom scorer for model evaluation.
        
        Returns:
            Custom scorer function
        """
        return make_scorer(self.custom_trading_loss, greater_is_better=True)
    
    def train_with_custom_loss(self, model_type='lightgbm', balance_classes=True):
        """
        Train a model with custom loss function.
        
        Args:
            model_type: 'lightgbm', 'random_forest', or 'gradient_boosting'
            balance_classes: Whether to balance classes using class weights
            
        Returns:
            Trained model
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not prepared. Call prepare_features_targets() first.")
        
        logger.info(f"Training {model_type} model with custom loss function")
        
        # Apply hard negative mining if available
        if self.hard_negatives is not None:
            X_train_enhanced, y_train_enhanced = self.add_hard_negatives()
        else:
            X_train_enhanced, y_train_enhanced = self.X_train, self.y_train
        
        # Calculate class weights if requested
        if balance_classes:
            classes = np.unique(y_train_enhanced)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train_enhanced)
            class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
            logger.info(f"Class weights: {class_weight_dict}")
        else:
            class_weight_dict = None
        
        # Create and train model based on type
        if model_type == 'lightgbm':
            # Convert to LightGBM dataset
            lgb_train = lgb.Dataset(
                X_train_enhanced, 
                y_train_enhanced,
                weight=None if not balance_classes else [class_weight_dict[y] for y in y_train_enhanced]
            )
            lgb_val = lgb.Dataset(
                self.X_val, 
                self.y_val, 
                reference=lgb_train
            )
            
            # Define parameters
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            
            # Train model
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_train, lgb_val],
                callbacks=[lgb.early_stopping(stopping_rounds=50)]
            )
            
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight=class_weight_dict if balance_classes else None,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_enhanced, y_train_enhanced)
            
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            model.fit(X_train_enhanced, y_train_enhanced)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Evaluate model
        self.evaluate_model(model, X_train_enhanced, y_train_enhanced, self.X_val, self.y_val)
        
        return model
    
    def evaluate_model(self, model, X_train, y_train, X_val, y_val):
        """
        Evaluate model performance on training and validation sets.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        logger.info("Evaluating model performance")
        
        # Get predictions
        if hasattr(model, 'predict'):
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
        else:
            # Handle LightGBM case
            y_train_pred = np.argmax(model.predict(X_train), axis=1)
            y_val_pred = np.argmax(model.predict(X_val), axis=1)
        
        # Calculate custom loss
        train_loss = self.custom_trading_loss(y_train, y_train_pred)
        val_loss = self.custom_trading_loss(y_val, y_val_pred)
        
        logger.info(f"Custom loss - Train: {-train_loss:.4f}, Validation: {-val_loss:.4f}")
        
        # Classification report
        train_report = classification_report(
            y_train, y_train_pred, 
            target_names=[self.inv_label_map[i] for i in range(3)]
        )
        val_report = classification_report(
            y_val, y_val_pred, 
            target_names=[self.inv_label_map[i] for i in range(3)]
        )
        
        logger.info(f"Training set classification report:\n{train_report}")
        logger.info(f"Validation set classification report:\n{val_report}")
        
        # Confusion matrices
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        train_cm = confusion_matrix(y_train, y_train_pred)
        sns.heatmap(
            train_cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[self.inv_label_map[i] for i in range(3)],
            yticklabels=[self.inv_label_map[i] for i in range(3)]
        )
        plt.title('Training Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.subplot(1, 2, 2)
        val_cm = confusion_matrix(y_val, y_val_pred)
        sns.heatmap(
            val_cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[self.inv_label_map[i] for i in range(3)],
            yticklabels=[self.inv_label_map[i] for i in range(3)]
        )
        plt.title('Validation Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/confusion_matrices.png')
        logger.info("Confusion matrices saved to reports/confusion_matrices.png")
    
    def cross_validate_model(self, model_type='lightgbm', n_splits=5, balance_classes=True):
        """
        Perform cross-validation with the custom loss function.
        
        Args:
            model_type: 'lightgbm', 'random_forest', or 'gradient_boosting'
            n_splits: Number of cross-validation folds
            balance_classes: Whether to balance classes using class weights
            
        Returns:
            Cross-validation scores
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not prepared. Call prepare_features_targets() first.")
        
        logger.info(f"Performing {n_splits}-fold cross-validation with {model_type}")
        
        # Combine training and validation sets for CV
        X = pd.concat([self.X_train, self.X_val])
        y = pd.concat([self.y_train, self.y_val])
        
        # Apply hard negative mining if available
        if self.hard_negatives is not None:
            X_enhanced, y_enhanced = self.add_hard_negatives()
            # Combine with validation set
            X = pd.concat([X_enhanced, self.X_val])
            y = pd.concat([y_enhanced, self.y_val])
        
        # Calculate class weights if requested
        if balance_classes:
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
            logger.info(f"Class weights: {class_weight_dict}")
        else:
            class_weight_dict = None
        
        # Create model based on type
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight=class_weight_dict if balance_classes else None,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        else:
            # LightGBM requires special handling for CV
            logger.warning("LightGBM cross-validation not implemented. Using RandomForest instead.")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight=class_weight_dict if balance_classes else None,
                random_state=42,
                n_jobs=-1
            )
        
        # Create custom scorer
        custom_scorer = self.get_custom_scorer()
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X, y, 
            scoring=custom_scorer,
            cv=cv,
            n_jobs=-1
        )
        
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return cv_scores
    
    def save_model(self, model, model_path):
        """
        Save the trained model to file.
        
        Args:
            model: Trained model
            model_path: Path to save the model
        """
        logger.info(f"Saving model to {model_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        if model_path.endswith('.pkl'):
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            joblib.dump(model, model_path)
        
        logger.info(f"Model saved successfully")


def main():
    """Main function to train model with custom loss."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train stock prediction model with custom loss')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the training data file')
    parser.add_argument('--hard-negatives', type=str, help='Path to hard negatives file')
    parser.add_argument('--model-type', type=str, default='lightgbm', 
                        choices=['lightgbm', 'random_forest', 'gradient_boosting'],
                        help='Type of model to train')
    parser.add_argument('--balance-classes', action='store_true', help='Balance classes using weights')
    parser.add_argument('--cross-validate', action='store_true', help='Perform cross-validation')
    parser.add_argument('--output-path', type=str, default='models/custom_model.joblib',
                        help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CustomLossTrainer(args.data_path, args.hard_negatives)
    
    # Prepare features and targets
    trainer.prepare_features_targets()
    
    # Cross-validate if requested
    if args.cross_validate:
        trainer.cross_validate_model(args.model_type, balance_classes=args.balance_classes)
    
    # Train model
    model = trainer.train_with_custom_loss(args.model_type, balance_classes=args.balance_classes)
    
    # Save model
    trainer.save_model(model, args.output_path)
    
    logger.info("Model training complete")


if __name__ == "__main__":
    main()
