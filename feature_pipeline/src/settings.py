'''import os

class Settings:
    RAW_DATA_PATH = '/workspaces/Machine-learning/data/raw/SundasPoDetail.xlsx'
    MODEL_DIR = '/workspaces/Machine-learning/models'
    AGGREGATION_LEVEL = 'weekly'
    MIN_HISTORICAL_DAYS = 1
    ML_MODELS = {
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1
        }
    }
    COMET_API_KEY = os.getenv('COMET_API_KEY')
    COMET_PROJECT_NAME = 'inventory-predictions'
    COMET_WORKSPACE = os.getenv('COMET_WORKSPACE_NAME')
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

settings = Settings()'''

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import logging

class FeaturePipelineConfig(BaseSettings):
    """Configuration for the feature pipeline."""
    data_path: str = str(Path("/workspaces/Machine-learning/data/raw/SundasPoDetail.xlsx"))
    output_path: str = str(Path("/workspaces/Machine-learning/data/transformed/monthly_features.parquet"))
    log_level: str = "INFO"
    aggregation_freq: str = "M"  # Monthly aggregation
    lag_periods: int = 2  # Number of lag features
    rolling_window: int = 3  # Rolling window for mean/std
    min_data_points: int = 3  # Minimum data points per SKU

    #class Config:
    #    env_file_encoding = "utf-8"

def setup_logging(log_level: str) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("feature_pipeline.log")
        ]
    )
