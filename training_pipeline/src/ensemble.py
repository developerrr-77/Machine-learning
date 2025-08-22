
import pandas as pd
import numpy as np
from typing import Dict, Optional
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """Combine Prophet and XGBoost predictions."""
    
    def __init__(self, prophet_model: Optional[Prophet], xgboost_model: Optional[XGBRegressor], weights: tuple = (0.7, 0.3)):
        self.prophet_model = prophet_model
        self.xgboost_model = xgboost_model
        self.weights = weights
    
    def predict(self, df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
        """Generate ensemble predictions."""
        try:
            prophet_preds, xgboost_preds = None, None
            
            if self.prophet_model:
                prophet_df = df[['OrderDate', 'NoOfPItems']].rename(columns={'OrderDate': 'ds', 'NoOfPItems': 'y'})
                future = self.prophet_model.make_future_dataframe(periods=periods, freq='ME')
                prophet_forecast = self.prophet_model.predict(future)
                prophet_preds = prophet_forecast['yhat'].iloc[-periods:]
            
            if self.xgboost_model:
                X = df[['Month', 'Year', 'Quarter', 'IsHoliday', 'Lag1', 'Lag2', 'RollingMean', 'RollingStd']]
                xgboost_preds = self.xgboost_model.predict(X.tail(periods))
            
            if prophet_preds is not None and xgboost_preds is not None:
                ensemble_preds = self.weights[0] * xgboost_preds + self.weights[1] * prophet_preds
            elif xgboost_preds is not None:
                ensemble_preds = xgboost_preds
            elif prophet_preds is not None:
                ensemble_preds = prophet_preds
            else:
                raise ValueError("Both models failed to generate predictions")
            
            return pd.DataFrame({
                'ds': future['ds'].iloc[-periods:] if self.prophet_model else pd.date_range(start=df['OrderDate'].iloc[-1], periods=periods+1, freq='ME')[1:],
                'yhat': ensemble_preds
            })
        
        except Exception as e:
            logger.error(f"Error generating ensemble predictions: {str(e)}")
            raise
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate ensemble model on test data."""
        try:
            prophet_preds, xgboost_preds = None, None
            
            if self.prophet_model:
                prophet_df = df[['OrderDate', 'NoOfPItems']].rename(columns={'OrderDate': 'ds', 'NoOfPItems': 'y'})
                prophet_forecast = self.prophet_model.predict(prophet_df)
                prophet_preds = prophet_forecast['yhat']
            
            if self.xgboost_model:
                X = df[['Month', 'Year', 'Quarter', 'IsHoliday', 'Lag1', 'Lag2', 'RollingMean', 'RollingStd']]
                xgboost_preds = self.xgboost_model.predict(X)
            
            if prophet_preds is not None and xgboost_preds is not None:
                ensemble_preds = self.weights[0] * xgboost_preds + self.weights[1] * prophet_preds
            elif xgboost_preds is not None:
                ensemble_preds = xgboost_preds
            elif prophet_preds is not None:
                ensemble_preds = prophet_preds
            else:
                raise ValueError("Both models failed to generate predictions")
            
            metrics = {
                'mae': mean_absolute_error(df['NoOfPItems'], ensemble_preds),
                'rmse': np.sqrt(mean_squared_error(df['NoOfPItems'], ensemble_preds))
            }
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {str(e)}")
            raise
