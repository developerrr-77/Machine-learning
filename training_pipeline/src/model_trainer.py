import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from comet_ml import Experiment  # Import comet_ml first
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import joblib
from pathlib import Path
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train Prophet and XGBoost models for a single SKU."""
    
    def __init__(self, n_splits: int = 3, test_size: float = 0.2, experiment: Optional[Experiment] = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.experiment = experiment
    
    def train_prophet(self, df: pd.DataFrame) -> Tuple[Optional[Prophet], Optional[Dict[str, float]]]:
        """Train Prophet model and evaluate."""
        try:
            prophet_df = df[['OrderDate', 'NoOfPItems']].rename(columns={'OrderDate': 'ds', 'NoOfPItems': 'y'})
            
            # Check if data is sufficient for cross-validation and has enough non-NaN rows
            if len(prophet_df) <= self.n_splits:
                logger.warning(f"Skipping Prophet training for SKU with {len(prophet_df)} samples, need more than {self.n_splits} for {self.n_splits}-fold CV")
                return None, None
            if prophet_df['y'].notna().sum() < 2:
                logger.warning(f"Skipping Prophet training for SKU with {prophet_df['y'].notna().sum()} non-NaN rows, need at least 2")
                return None, None
            
            # Dynamically set n_changepoints based on data size
            n_changepoints = min(25, len(prophet_df) // 2)
            
            # Ensure test_size is at least 1
            test_size = max(1, int(len(df) * self.test_size))
            tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=test_size)
            metrics = {'mae': [], 'rmse': []}
            
            for train_idx, test_idx in tscv.split(prophet_df):
                train_df, test_df = prophet_df.iloc[train_idx], prophet_df.iloc[test_idx]
                if train_df['y'].notna().sum() < 2:
                    logger.warning(f"Skipping CV fold: train_df has {train_df['y'].notna().sum()} non-NaN rows, need at least 2")
                    continue
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    n_changepoints=n_changepoints
                )
                model.fit(train_df)
                future = model.make_future_dataframe(periods=len(test_df), freq='ME')
                forecast = model.predict(future)
                preds = forecast['yhat'].iloc[-len(test_df):]
                metrics['mae'].append(mean_absolute_error(test_df['y'], preds))
                metrics['rmse'].append(np.sqrt(mean_squared_error(test_df['y'], preds)))
            
            if not metrics['mae']:
                logger.warning("No valid CV folds processed for Prophet due to insufficient non-NaN data")
                return None, None
            
            # Train on full data
            if prophet_df['y'].notna().sum() < 2:
                logger.warning(f"Skipping final Prophet training: {prophet_df['y'].notna().sum()} non-NaN rows, need at least 2")
                return None, None
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                n_changepoints=n_changepoints
            )
            model.fit(prophet_df)
            metrics = {'mae': np.mean(metrics['mae']), 'rmse': np.mean(metrics['rmse'])}
            logger.info(f"Prophet trained, metrics: {metrics}")
            return model, metrics
        
        except Exception as e:
            logger.error(f"Error training Prophet: {str(e)}")
            if self.experiment:
                self.experiment.log_other("prophet_error", str(e))
            raise
    
    def objective_xgb(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, tscv: TimeSeriesSplit) -> float:
        """Optuna objective function for XGBoost hyperparameter tuning."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
        }
        model = XGBRegressor(**params, objective='reg:squarederror')
        mae_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if y_train.notna().sum() < 2:
                logger.warning(f"Skipping CV fold: y_train has {y_train.notna().sum()} non-NaN rows, need at least 2")
                continue
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae_scores.append(mean_absolute_error(y_test, preds))
        
        if not mae_scores:
            logger.warning("No valid CV folds processed for XGBoost due to insufficient non-NaN data")
            return float('inf')
        return np.mean(mae_scores)
    
    def train_xgboost(self, df: pd.DataFrame) -> Tuple[Optional[XGBRegressor], Optional[Dict[str, float]]]:
        """Train XGBoost model with Optuna tuning and evaluate."""
        try:
            X = df[['Month', 'Year', 'Quarter', 'IsHoliday', 'Lag1', 'Lag2', 'RollingMean', 'RollingStd']]
            y = df['NoOfPItems']
            
            if len(df) <= self.n_splits:
                logger.warning(f"Skipping XGBoost training for SKU with {len(df)} samples, need more than {self.n_splits} for {self.n_splits}-fold CV")
                return None, None
            if y.notna().sum() < 2:
                logger.warning(f"Skipping XGBoost training for SKU with {y.notna().sum()} non-NaN rows, need at least 2")
                return None, None
            
            test_size = max(1, int(len(df) * self.test_size))
            tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=test_size)
            
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self.objective_xgb(trial, X, y, tscv), n_trials=20)
            best_params = study.best_params
            logger.info(f"Best XGBoost params: {best_params}")
            
            model = XGBRegressor(**best_params, objective='reg:squarederror')
            metrics = {'mae': [], 'rmse': []}
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                if y_train.notna().sum() < 2:
                    logger.warning(f"Skipping CV fold: y_train has {y_train.notna().sum()} non-NaN rows, need at least 2")
                    continue
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                metrics['mae'].append(mean_absolute_error(y_test, preds))
                metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, preds)))
            
            if not metrics['mae']:
                logger.warning("No valid CV folds processed for XGBoost due to insufficient non-NaN data")
                return None, None
            
            if y.notna().sum() < 2:
                logger.warning(f"Skipping final XGBoost training: {y.notna().sum()} non-NaN rows, need at least 2")
                return None, None
            model.fit(X, y)
            metrics = {'mae': np.mean(metrics['mae']), 'rmse': np.mean(metrics['rmse'])}
            logger.info(f"XGBoost trained, metrics: {metrics}")
            return model, metrics
        
        except Exception as e:
            logger.error(f"Error training XGBoost: {str(e)}")
            if self.experiment:
                self.experiment.log_other("xgboost_error", str(e))
            raise
    
    def save_model(self, model: object, model_type: str, sku: str, model_dir: str) -> None:
        """Save model to disk and push to Comet.ml Model Registry."""
        try:
            # Sanitize SKU name
            safe_sku = re.sub(r'[^\w\-]', '_', sku.strip())
            # Create unique model name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{safe_sku}_{model_type}_{timestamp}"
            
            # Save to disk
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            model_path = Path(model_dir) / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_type} model for SKU {sku} to {model_path}")
            
            # Push to Comet.ml Model Registry
            if self.experiment:
                self.experiment.log_model(model_name, str(model_path), file_name=f"{model_name}.pkl")
                logger.info(f"Pushed {model_type} model for SKU {sku} to Comet.ml Model Registry as {model_name}")
        
        except Exception as e:
            logger.error(f"Error saving or pushing {model_type} model for SKU {sku}: {str(e)}")
            if self.experiment:
                self.experiment.log_other(f"{model_type}_save_error", str(e))
            raise