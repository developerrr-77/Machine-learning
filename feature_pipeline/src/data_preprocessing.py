
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """Clean and preprocess data."""
    
    def __init__(self, aggregation_freq: str = "M"):
        self.aggregation_freq = aggregation_freq
    
    def clean_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Clean data and aggregate to specified frequency."""
        try:
            # Remove duplicates
            initial_rows = df.shape[0]
            df = df.drop_duplicates()
            logger.info(f"Removed {initial_rows - df.shape[0]} duplicate rows")
            
            # Check for missing values
            if df.isna().any().any():
                logger.warning("Found missing values in data")
                df = df.dropna()
                logger.info(f"Dropped rows with missing values, new shape: {df.shape}")
            
            # Aggregate to monthly demand per SKU
            df_agg = df.groupby(['SkuName', 'OrderDate'])['NoOfPItems'].sum().reset_index()
            monthly_demand = df_agg.set_index('OrderDate').groupby('SkuName')['NoOfPItems'].resample(self.aggregation_freq).sum().reset_index()
            
            logger.info(f"Aggregated data to {self.aggregation_freq} frequency, shape: {monthly_demand.shape}")
            return monthly_demand
        
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
