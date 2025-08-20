
import pandas as pd
import numpy as np
from typing import Dict
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)

class Ensemble:
    """Combine Prophet and XGBoost predictions for demand forecasting."""
    
    def evaluate(self, prophet_model: Prophet, xgboost_model: XGBRegressor, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate ensemble of Prophet and XGBoost models."""
        try:
            # Prepare data for Prophet
            prophet_df = df[['OrderDate', 'NoOfPItems']].rename(columns={'OrderDate': 'ds', 'NoOfPItems': 'y'})
            
            # Generate Prophet predictions
            future = prophet_model.make_future_dataframe(periods=0, freq='ME')
            prophet_forecast = prophet_model.predict(future)
            prophet_preds = prophet_forecast['yhat'].values
            
            # Prepare data for XGBoost
            X = df[['Month', 'Year', 'Quarter', 'IsHoliday', 'Lag1', 'Lag2', 'RollingMean', 'RollingStd']]
            y = df['NoOfPItems']
            
            # Generate XGBoost predictions
            xgboost_preds = xgboost_model.predict(X)
            
            # Combine predictions (simple 50/50 weighted average)
            ensemble_preds = 0.5 * prophet_preds + 0.5 * xgboost_preds
            
            # Compute metrics
            metrics = {
                'mae': mean_absolute_error(y, ensemble_preds),
                'rmse': np.sqrt(mean_squared_error(y, ensemble_preds))
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {str(e)}")
            raise