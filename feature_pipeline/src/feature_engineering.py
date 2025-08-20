
import pandas as pd
import holidays
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Generate features for time-series forecasting."""
    
    def __init__(self, lag_periods: int = 2, rolling_window: int = 3, country: str = "PK"):
        self.lag_periods = lag_periods
        self.rolling_window = rolling_window
        self.holidays = holidays.CountryHoliday(country)
    
    def add_features(self, df: pd.DataFrame, sku: str) -> Optional[pd.DataFrame]:
        """Add features for a single SKU."""
        try:
            item_df = df[df['SkuName'] == sku].sort_values('OrderDate')
            if len(item_df) < self.lag_periods + 1:
                logger.warning(f"Skipping SKU {sku}: insufficient data points ({len(item_df)})")
                return None
            
            # Time-based features
            item_df['Month'] = item_df['OrderDate'].dt.month
            item_df['Year'] = item_df['OrderDate'].dt.year
            item_df['Quarter'] = item_df['OrderDate'].dt.quarter
            item_df['IsHoliday'] = item_df['OrderDate'].apply(lambda x: 1 if x in self.holidays else 0)
            
            # Lag features
            for i in range(1, self.lag_periods + 1):
                item_df[f'Lag{i}'] = item_df['NoOfPItems'].shift(i).fillna(0)
            
            # Rolling statistics
            item_df['RollingMean'] = item_df['NoOfPItems'].rolling(window=self.rolling_window).mean().fillna(0)
            item_df['RollingStd'] = item_df['NoOfPItems'].rolling(window=self.rolling_window).std().fillna(0)
            
            logger.info(f"Engineered features for SKU {sku}, shape: {item_df.shape}")
            return item_df
        
        except Exception as e:
            logger.error(f"Error engineering features for SKU {sku}: {str(e)}")
            raise
