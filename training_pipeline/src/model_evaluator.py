
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate ensemble of Prophet and XGBoost models."""
    
    def __init__(self, n_splits: int = 3, test_size: float = 0.2):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def evaluate_ensemble(self, prophet_model: Prophet, xgb_model: XGBRegressor, 
                         df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate ensemble predictions."""
        try:
            prophet_df = df[['OrderDate', 'NoOfPItems']].rename(columns={'OrderDate': 'ds', 'NoOfPItems': 'y'})
            X = df[['Month', 'Year', 'Quarter', 'IsHoliday', 'Lag1', 'Lag2', 'RollingMean', 'RollingStd']]
            y = df['NoOfPItems']
            tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=int(len(df) * self.test_size))
            
            metrics = {'mae': [], 'rmse': []}
            for train_idx, test_idx in tscv.split(df):
                prophet_train, prophet_test = prophet_df.iloc[train_idx], prophet_df.iloc[test_idx]
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_test = y.iloc[test_idx]
                
                # Prophet predictions
                future = prophet_model.make_future_dataframe(periods=len(prophet_test), freq='ME')
                prophet_preds = prophet_model.predict(future)['yhat'].iloc[-len(prophet_test):]
                
                # XGBoost predictions
                xgb_preds = xgb_model.predict(X_test)
                
                # Ensemble: average
                ensemble_preds = (prophet_preds + xgb_preds) / 2
                metrics['mae'].append(mean_absolute_error(y_test, ensemble_preds))
                metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, ensemble_preds)))
            
            metrics = {'mae': np.mean(metrics['mae']), 'rmse': np.mean(metrics['rmse'])}
            logger.info(f"Ensemble metrics: {metrics}")
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {str(e)}")
            raise
