
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and parse Excel data."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """Load Excel file and convert serial dates to datetime if necessary."""
        try:
            if not self.data_path.exists():
                logger.error(f"Data file not found: {self.data_path}")
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            df = pd.read_excel(self.data_path, engine="openpyxl")
            logger.info(f"Loaded data from {self.data_path}, shape: {df.shape}")
            
            # Validate required columns
            required_columns = ['PONumber', 'OrderDate', 'SkuName', 'SupplierName', 'NoOfPItems']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                logger.error(f"Missing required columns: {missing}")
                raise ValueError(f"Missing required columns: {missing}")
            
            # Convert NoOfPItems to numeric
            df['NoOfPItems'] = pd.to_numeric(df['NoOfPItems'], errors='coerce')
            if df['NoOfPItems'].isna().any():
                logger.warning("Found NaN values in NoOfPItems after conversion")
                df = df.dropna(subset=['NoOfPItems'])
                logger.info(f"Dropped rows with NaN in NoOfPItems, new shape: {df.shape}")
            
            # Handle OrderDate conversion
            if pd.api.types.is_numeric_dtype(df['OrderDate']):
                logger.info("OrderDate is numeric, converting serial dates to datetime")
                df['OrderDate'] = df['OrderDate'].apply(
                    lambda x: datetime(1899, 12, 30) + timedelta(days=int(x))
                ).astype("datetime64[ns]")
            elif pd.api.types.is_datetime64_any_dtype(df['OrderDate']):
                logger.info("OrderDate is already in datetime format")
                df['OrderDate'] = df['OrderDate'].astype("datetime64[ns]")
            else:
                logger.error("OrderDate column has unsupported format")
                raise ValueError("OrderDate column must be numeric (serial dates) or datetime")
            
            logger.info("Data loaded and parsed successfully")
            return df
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
