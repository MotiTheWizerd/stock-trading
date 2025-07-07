"""
Model Evaluation and Backtesting Module
=======================================

This module provides comprehensive evaluation capabilities for trading models,
including:

1. Performance metrics (accuracy, precision, recall, F1)
2. Signal confidence analysis
3. PnL simulation and backtesting
4. Risk analysis (drawdown, Sharpe ratio)
5. Visualization and reporting

Usage:
    from model_evaluator import ModelEvaluator
    
    evaluator = ModelEvaluator('AAPL')
    evaluator.load_model('models/AAPL/model_AAPL_20231207.joblib')
    results = evaluator.evaluate_model(test_data)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from paths import get_data_path, get_models_path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and backtesting."""
    
    def __init__(self, ticker: str):
        """Initialize evaluator for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
        """
        self.ticker = ticker.upper()
        self.data_path = get_data_path(ticker)
        self.models_path = get_models_path(ticker)
        self.model = None
        self.model_metadata = None
        
    def load_model(self, model_path: Optional[Path] = None) -> None:
        """Load a trained model for evaluation.
        
        Args:
            model_path: Path to model file. If None, loads latest model.
        """
        if model_path is None:
            model_path = self._get_latest_model_path()
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Load metadata if available
        metadata_path = model_path.parent / f"metadata_{model_path.stem.split('_', 1)[1]}.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
        
        logger.info(f"📥 Loaded model: {model_path}")
    
    def _get_latest_model_path(self) -> Path:
        """Get the path to the latest model file."""
        model_files = list(self.models_path.glob(f"model_{self.ticker}_*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model files found for {self.ticker}")
        
        # Sort by timestamp in filename
        model_files.sort(key=lambda x: x.stem.split('_')[-1])
        return model_files[-1]
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate model performance on test data.
        
        Args:
            test_data: DataFrame with features and target
            
        Returns:
            Dictionary with evaluation results
        """
        if self.model is None:
            raise ValueError("Model must be loaded before evaluation")
        
        logger.info("📊 Evaluating model performance")
        
        # Prepare features and target
        X_test, y_test = self._prepare_test_data(test_data)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        
        # Analyze signals
        signal_analysis = self._analyze_signals(y_test, y_pred, y_proba)
        
        # Confidence analysis
        confidence_analysis = self._analyze_confidence(y_proba)
        
        results = {
            'metrics': metrics,
            'signal_analysis': signal_analysis,
            'confidence_analysis': confidence_analysis,
            'predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_proba': y_proba.tolist()
            }
        }
        
        return results
    
    def _prepare_test_data(self, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare test data for evaluation."""
        # Get feature columns from metadata if available
        if self.model_metadata and 'feature_columns' in self.model_metadata:
            feature_cols = self.model_metadata['feature_columns']
        else:
            # Default feature selection
            exclude_cols = {"datetime", "ticker", "signal"}
            feature_cols = [col for col in test_data.columns if col not in exclude_cols]
        
        X_test = test_data[feature_cols].copy()
        y_test = test_data["signal"].fillna("NONE")
        
        # Handle missing values
        X_test = X_test.fillna(method='ffill').fillna(method='bfill')
        
        return X_test, y_test
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Calculate various performance metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        class_report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist()
        }
    
    def _analyze_signals(self, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Analyze signal distribution and success rates."""
        # Signal distribution
        true_dist = y_true.value_counts(normalize=True).to_dict()
        pred_dist = pd.Series(y_pred).value_counts(normalize=True).to_dict()
        
        # Success rate by signal type
        success_rates = {}
        for signal in np.unique(y_true):
            mask = y_true == signal
            if mask.sum() > 0:
                success_rates[signal] = accuracy_score(y_true[mask], y_pred[mask])
        
        return {
            'true_distribution': true_dist,
            'predicted_distribution': pred_dist,
            'success_rates': success_rates
        }
    
    def _analyze_confidence(self, y_proba: np.ndarray) -> Dict:
        """Analyze prediction confidence."""
        max_proba = np.max(y_proba, axis=1)
        
        # Confidence statistics
        confidence_stats = {
            'mean': np.mean(max_proba),
            'std': np.std(max_proba),
            'min': np.min(max_proba),
            'max': np.max(max_proba),
            'median': np.median(max_proba),
            'q25': np.percentile(max_proba, 25),
            'q75': np.percentile(max_proba, 75)
        }
        
        # Confidence bins
        bins = np.linspace(0, 1, 11)
        confidence_bins = np.histogram(max_proba, bins=bins)[0].tolist()
        
        return {
            'statistics': confidence_stats,
            'histogram': confidence_bins,
            'bin_edges': bins.tolist()
        }
    
    def run_backtest(self, test_data: pd.DataFrame, initial_balance: float = 10000) -> Dict:
        """Run backtesting simulation with PnL calculation.
        
        Args:
            test_data: DataFrame with OHLCV data and signals
            initial_balance: Starting balance for simulation
            
        Returns:
            Dictionary with backtest results
        """
        if self.model is None:
            raise ValueError("Model must be loaded before backtesting")
        
        logger.info("🔄 Running backtest simulation")
        
        # Prepare data
        X_test, y_true = self._prepare_test_data(test_data)
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Add predictions to test data
        backtest_data = test_data.copy()
        backtest_data['predicted_signal'] = y_pred
        backtest_data['confidence'] = np.max(y_proba, axis=1)
        
        # Simulate trading
        portfolio = self._simulate_trading(backtest_data, initial_balance)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(portfolio)
        
        # Risk analysis
        risk_metrics = self._calculate_risk_metrics(portfolio)
        
        results = {
            'portfolio': portfolio,
            'performance': performance,
            'risk_metrics': risk_metrics,
            'final_balance': portfolio['balance'].iloc[-1],
            'total_return': (portfolio['balance'].iloc[-1] - initial_balance) / initial_balance,
            'num_trades': len(portfolio[portfolio['position_change'] != 0])
        }
        
        return results
    
    def _simulate_trading(self, data: pd.DataFrame, initial_balance: float) -> pd.DataFrame:
        """Simulate trading based on model predictions."""
        portfolio = data.copy()
        portfolio['position'] = 0  # 0: cash, 1: long, -1: short
        portfolio['shares'] = 0.0
        portfolio['cash'] = initial_balance
        portfolio['balance'] = initial_balance
        portfolio['position_change'] = 0
        
        position = 0
        shares = 0.0
        cash = initial_balance
        
        for i in range(len(portfolio)):
            signal = portfolio.iloc[i]['predicted_signal']
            confidence = portfolio.iloc[i]['confidence']
            price = portfolio.iloc[i]['Close']
            
            # Trading logic with confidence threshold
            if confidence > 0.6:  # Only trade with high confidence
                if signal == 'BUY' and position <= 0:
                    # Buy signal
                    shares = cash / price
                    cash = 0
                    position = 1
                    portfolio.iloc[i, portfolio.columns.get_loc('position_change')] = 1
                    
                elif signal == 'SELL' and position >= 0:
                    # Sell signal
                    cash = shares * price
                    shares = 0
                    position = -1
                    portfolio.iloc[i, portfolio.columns.get_loc('position_change')] = -1
            
            # Update portfolio state
            portfolio.iloc[i, portfolio.columns.get_loc('position')] = position
            portfolio.iloc[i, portfolio.columns.get_loc('shares')] = shares
            portfolio.iloc[i, portfolio.columns.get_loc('cash')] = cash
            
            # Calculate total balance
            balance = cash + shares * price
            portfolio.iloc[i, portfolio.columns.get_loc('balance')] = balance
        
        return portfolio
    
    def _calculate_performance_metrics(self, portfolio: pd.DataFrame) -> Dict:
        """Calculate portfolio performance metrics."""
        returns = portfolio['balance'].pct_change().dropna()
        
        # Basic performance
        total_return = (portfolio['balance'].iloc[-1] - portfolio['balance'].iloc[0]) / portfolio['balance'].iloc[0]
        
        # Annualized return (assuming daily data)
        num_days = len(portfolio)
        annualized_return = (1 + total_return) ** (365 / num_days) - 1
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Win rate
        winning_trades = returns[returns > 0]
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_return': returns.mean(),
            'max_return': returns.max(),
            'min_return': returns.min()
        }
    
    def _calculate_risk_metrics(self, portfolio: pd.DataFrame) -> Dict:
        """Calculate risk metrics."""
        returns = portfolio['balance'].pct_change().dropna()
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (95% confidence)
        var_95 = returns.quantile(0.05)
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252)
        sortino_ratio = (returns.mean() * 252) / downside_std if downside_std > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'sortino_ratio': sortino_ratio,
            'downside_volatility': downside_std
        }
    
    def create_evaluation_report(self, evaluation_results: Dict, backtest_results: Dict) -> str:
        """Create a comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_model
            backtest_results: Results from run_backtest
            
        Returns:
            HTML report string
        """
        metrics = evaluation_results['metrics']
        performance = backtest_results['performance']
        risk = backtest_results['risk_metrics']
        
        report = f"""
        <html>
        <head>
            <title>Model Evaluation Report - {self.ticker}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
                .highlight {{ background-color: #f0f8ff; padding: 10px; border-left: 4px solid #007acc; }}
            </style>
        </head>
        <body>
            <h1>Model Evaluation Report: {self.ticker}</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Model Performance</h2>
                <div class="metric">Accuracy: {metrics['accuracy']:.4f}</div>
                <div class="metric">Precision: {metrics['precision']:.4f}</div>
                <div class="metric">Recall: {metrics['recall']:.4f}</div>
                <div class="metric">F1 Score: {metrics['f1_score']:.4f}</div>
            </div>
            
            <div class="section">
                <h2>Trading Performance</h2>
                <div class="highlight">
                    <div class="metric">Total Return: {performance['total_return']:.2%}</div>
                    <div class="metric">Annualized Return: {performance['annualized_return']:.2%}</div>
                    <div class="metric">Sharpe Ratio: {performance['sharpe_ratio']:.2f}</div>
                </div>
                <div class="metric">Win Rate: {performance['win_rate']:.2%}</div>
                <div class="metric">Volatility: {performance['volatility']:.2%}</div>
            </div>
            
            <div class="section">
                <h2>Risk Metrics</h2>
                <div class="metric">Maximum Drawdown: {risk['max_drawdown']:.2%}</div>
                <div class="metric">Sortino Ratio: {risk['sortino_ratio']:.2f}</div>
                <div class="metric">VaR (95%): {risk['var_95']:.2%}</div>
            </div>
            
            <div class="section">
                <h2>Trading Summary</h2>
                <div class="metric">Final Balance: ${backtest_results['final_balance']:.2f}</div>
                <div class="metric">Number of Trades: {backtest_results['num_trades']}</div>
            </div>
        </body>
        </html>
        """
        
        return report
    
    def save_evaluation_report(self, evaluation_results: Dict, backtest_results: Dict) -> Path:
        """Save evaluation report to file.
        
        Args:
            evaluation_results: Results from evaluate_model
            backtest_results: Results from run_backtest
            
        Returns:
            Path to saved report
        """
        report_html = self.create_evaluation_report(evaluation_results, backtest_results)
        
        # Create reports directory
        reports_dir = self.models_path / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"evaluation_report_{self.ticker}_{timestamp}.html"
        
        with open(report_path, 'w') as f:
            f.write(report_html)
        
        logger.info(f"📊 Evaluation report saved: {report_path}")
        return report_path


def main():
    """Example usage of ModelEvaluator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Evaluation Tool")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    parser.add_argument("--model-path", type=str, help="Path to model file")
    parser.add_argument("--test-days", type=int, default=30, help="Number of days to use for testing")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.ticker)
    
    # Load model
    if args.model_path:
        evaluator.load_model(Path(args.model_path))
    else:
        evaluator.load_model()
    
    # Load test data (this would need to be implemented based on your data structure)
    # For now, this is a placeholder
    print(f"Model evaluator initialized for {args.ticker}")
    print("Test data loading and evaluation would be implemented here")


if __name__ == "__main__":
    main()
