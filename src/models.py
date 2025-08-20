'''import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from prophet import Prophet
import joblib

class BaseModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseModel':
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
    
    def save_model(self, filepath: str) -> None:
        raise NotImplementedError

class XGBoostModel(BaseModel):
    def __init__(self, model_name: str = "XGBoost"):
        super().__init__(model_name)
        self.model = XGBRegressor(random_state=42)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'XGBoostModel':
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        joblib.dump(self.model, filepath)
        self.logger.info(f"XGBoost model saved to {filepath}")

class ProphetModel(BaseModel):
    def __init__(self, model_name: str = "Prophet"):
        super().__init__(model_name)
        self.model = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'ProphetModel':
        try:
            df = pd.DataFrame({
                'ds': pd.to_datetime(X['date'], errors='coerce'),
                'y': y.values
            })
            df = df.dropna()
            if len(df) < 6:
                raise ValueError(f"Insufficient valid data for Prophet: {len(df)} records")
            self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            self.model.add_country_holidays(country_name='PK')
            self.model.fit(df)
            self.is_fitted = True
        except Exception as e:
            self.logger.warning(f"Prophet fit failed: {e}. Using mean prediction.")
            self.model = None
            self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), X['demand'].mean() if 'demand' in X.columns else 0)
        try:
            future = pd.DataFrame({
                'ds': pd.to_datetime(X.index.get_level_values('date'), errors='coerce')
            })
            future = future.dropna()
            if len(future) == 0:
                return np.full(len(X), X['demand'].mean() if 'demand' in X.columns else 0)
            forecast = self.model.predict(future)
            return np.array(forecast['yhat'])
        except:
            self.logger.warning("Prophet predict failed. Using mean prediction.")
            return np.full(len(X), X['demand'].mean() if 'demand' in X.columns else 0)
    
    def save_model(self, filepath: str) -> None:
        if self.model is not None:
            joblib.dump(self, filepath)
            self.logger.info(f"Prophet model saved to {filepath}")

class ModelTrainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def train_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, optimize_hyperparams: bool = False, hyperparameters: Dict[str, Any] = None) -> tuple:
        if model_type == 'xgboost':
            model = XGBoostModel()
            if optimize_hyperparams and hyperparameters:
                search = RandomizedSearchCV(
                    estimator=model.model,
                    param_distributions=hyperparameters,
                    n_iter=10,
                    cv=3,
                    scoring='neg_mean_absolute_error',
                    random_state=42,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)
                model.model = search.best_estimator_
                self.logger.info(f"Best hyperparameters for {model.model_name}: {search.best_params_}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return model, {'mae': mean_absolute_error(y_val, y_pred), 'rmse': np.sqrt(mean_squared_error(y_val, y_pred))}
        elif model_type == 'prophet':
            model = ProphetModel()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return model, {'mae': mean_absolute_error(y_val, y_pred), 'rmse': np.sqrt(mean_squared_error(y_val, y_pred))}
        else:
            raise ValueError(f"Unsupported model type: {model_type}")'''

import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from comet_ml import Experiment
import numpy as np
import logging
import joblib

logger = logging.getLogger(__name__)

def calculate_mape(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-10, y_true)  # Avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_and_evaluate(df, product, experiment):
    df_product = df[df['sku_name'] == product].copy()
    df_product = df_product.sort_values('date')
    prophet_df = df_product[['date', 'demand']].rename(columns={'date': 'ds', 'demand': 'y'})
    train_size = int(0.8 * len(prophet_df))
    train_df, test_df = prophet_df[:train_size], prophet_df[train_size:]

    # Prophet model
    model = Prophet(changepoint_prior_scale=0.01, seasonality_prior_scale=0.1)
    try:
        model.fit(train_df)
        future = model.make_future_dataframe(periods=len(test_df))
        forecast = model.predict(future)
        y_pred = forecast['yhat'].tail(len(test_df))
    except:
        logger.warning(f"Prophet failed for {product}. Using XGBoost.")
        features = ['demand_lag_1', 'demand_lag_3', 'demand_lag_7', 'demand_rolling_mean_7', 'demand_rolling_std_7', 'week_sin', 'week_cos', 'month_sin', 'month_cos', 'is_weekend', 'is_holiday', 'is_medical', 'supplier_encoded']
        X_train = train_df[features].fillna(0)
        y_train = train_df['y']
        X_test = test_df[features].fillna(0)
        y_test = test_df['y']

        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        model = xgb_model  # Save XGBoost if Prophet fails

    if y_pred is None or np.any(np.isnan(y_pred)):
        logger.warning(f"XGBoost failed for {product}. Using mean prediction.")
        y_pred = np.full(len(test_df), train_df['y'].mean())

    mae = mean_absolute_error(test_df['y'], y_pred)
    rmse = np.sqrt(mean_squared_error(test_df['y'], y_pred))
    mape = calculate_mape(test_df['y'], y_pred)

    experiment.log_metrics({f'{product}_mae': mae, f'{product}_rmse': rmse, f'{product}_mape': mape})

    joblib.dump(model, f'models/{product}_model.pkl')
    logger.info(f"Model saved for {product}")
    return mae, rmse, mape