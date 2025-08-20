
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import logging
import os 
from dotenv import load_dotenv
load_dotenv()

class TrainingPipelineConfig(BaseSettings):
    """Configuration for the training pipeline."""
    feature_path: str = str(Path("/workspaces/Machine-learning/data/transformed/monthly_features.parquet"))
    model_dir: str = str(Path("training_pipeline/models"))
    log_level: str = "INFO"
    test_size: float = 0.2  # Test split proportion
    n_splits: int = 3  # TimeSeriesSplit folds
    min_data_points: int = 3  # Minimum data points per SKU
    comet_api_key: Optional[str] = os.getenv("COMET_API_KEY")  # Comet ML API key
    comet_project_name: Optional[str] = "inventory-prediction"  # Comet ML project name
    hopsworks_api_key: Optional[str] = os.getenv("HOPSWORKS_API_KEY")  # Hopsworks API key
    hopsworks_project_name: Optional[str] = os.getenv("HOPSWORKS_PROJECT_NAME")  # Hopsworks project name

def setup_logging(log_level: str) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training_pipeline.log")
        ]
    )
