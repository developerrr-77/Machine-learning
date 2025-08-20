'''import pandas as pd
import numpy as np
import logging
from typing import List
from sklearn.ensemble import RandomForestRegressor

class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_all_features(self, df: pd.DataFrame, target_col: str = 'demand') -> pd.DataFrame:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        for lag in [1, 2]:
            df[f'demand_lag_{lag}'] = df.groupby('product_id')[target_col].shift(lag)
        
        df['demand_rolling_mean_2'] = df.groupby('product_id')[target_col].shift(1).rolling(window=2, min_periods=1).mean()
        df['demand_rolling_std_2'] = df.groupby('product_id')[target_col].shift(1).rolling(window=2, min_periods=1).std()
        
        df = df.fillna(0)
        self.logger.info(f"Feature engineered data shape: {df.shape}")
        return df
    
    def select_important_features(self, df: pd.DataFrame, target_col: str = 'demand', max_features: int = 18) -> List[str]:
        feature_cols = [col for col in df.columns if col not in ['product_id', 'date', target_col, 'order_count', 'supplier']]
        if len(feature_cols) <= max_features:
            return feature_cols
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        feature_importance = pd.Series(model.feature_importances_, index=feature_cols)
        selected_features = feature_importance.nlargest(max_features).index.tolist()
        self.logger.info(f"Selected features: {selected_features}")
        return selected_features'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import holidays
import logging

logger = logging.getLogger(__name__)

def create_features(df):
    df = df.copy()

    # Lagged features
    for lag in [1, 3, 7]:
        df[f'demand_lag_{lag}'] = df.groupby('sku_name')['demand'].shift(lag)

    # Rolling stats
    df['demand_rolling_mean_7'] = df.groupby('sku_name')['demand'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    df['demand_rolling_std_7'] = df.groupby('sku_name')['demand'].rolling(7, min_periods=1).std().reset_index(0, drop=True)

    # Temporal features
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.isocalendar().week
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Holidays (Pakistan)
    pk_holidays = holidays.Pakistan(years=[df['date'].dt.year.min(), df['date'].dt.year.max()])
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in pk_holidays else 0)

    # Product type
    medical_keywords = ['Syringes', 'Anti', 'Factor', 'Diluent', 'Lyse', 'Injection']
    df['is_medical'] = df['sku_name'].apply(lambda x: 1 if any(kw in x for kw in medical_keywords) else 0)

    # Supplier encoding
    le = LabelEncoder()
    df['supplier_encoded'] = le.fit_transform(df['supplier_name'])

    # Clustering
    variance = df.groupby('sku_name')['demand'].var().reset_index()
    kmeans = KMeans(n_clusters=3)
    variance['cluster'] = kmeans.fit_predict(variance[['demand']].fillna(0))
    df = df.merge(variance[['sku_name', 'cluster']], on='sku_name')

    df = df.fillna(0)
    logger.info("Features created")
    return df