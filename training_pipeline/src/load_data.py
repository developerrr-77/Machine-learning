
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Load feature data from Parquet."""
    
    def __init__(self, feature_path: str):
        self.feature_path = Path(feature_path)
    
    def load_features(self) -> Optional[pd.DataFrame]:
        """Load feature Parquet file."""
        try:
            if not self.feature_path.exists():
                logger.error(f"Feature file not found: {self.feature_path}")
                raise FileNotFoundError(f"Feature file not found: {self.feature_path}")
            
            df = pd.read_parquet(self.feature_path, engine="pyarrow")
            logger.info(f"Loaded features from {self.feature_path}, shape: {df.shape}")
            
            # Validate required columns
            required_columns = ['SkuName', 'OrderDate', 'NoOfPItems', 'Month', 'Year', 'Quarter', 
                               'IsHoliday', 'Lag1', 'Lag2', 'RollingMean', 'RollingStd']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                logger.error(f"Missing required columns: {missing}")
                raise ValueError(f"Missing required columns: {missing}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            raise
