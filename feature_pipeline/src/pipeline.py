
import pandas as pd
from pathlib import Path
from typing import Optional
import logging
from settings import FeaturePipelineConfig, setup_logging
from data import DataLoader
from data_preprocessing import DataCleaner
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """Orchestrate the feature engineering pipeline."""
    
    def __init__(self, config: FeaturePipelineConfig):
        self.config = config
        self.loader = DataLoader(config.data_path)
        self.cleaner = DataCleaner(config.aggregation_freq)
        self.engineer = FeatureEngineer(config.lag_periods, config.rolling_window)
    
    def run(self) -> Optional[pd.DataFrame]:
        """Run the full pipeline and save output."""
        try:
            # Setup logging
            setup_logging(self.config.log_level)
            logger.info("Starting feature pipeline")
            
            # Load data
            df = self.loader.load_data()
            if df is None:
                logger.error("Failed to load data")
                return None
            
            # Clean and aggregate
            monthly_demand = self.cleaner.clean_data(df)
            if monthly_demand is None:
                logger.error("Failed to clean data")
                return None
            
            # Engineer features for each SKU
            all_features = []
            for sku in monthly_demand['SkuName'].unique():
                features = self.engineer.add_features(monthly_demand, sku)
                if features is not None:
                    all_features.append(features)
            
            if not all_features:
                logger.error("No SKUs had sufficient data for feature engineering")
                return None
            
            final_df = pd.concat(all_features, ignore_index=True)
            logger.info(f"Final feature DataFrame shape: {final_df.shape}")
            
            # Save to Parquet
            output_path = Path(self.config.output_path)
            final_df.to_parquet(output_path, engine='pyarrow', index=False)
            logger.info(f"Saved features to {output_path}")
            
            return final_df
        
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise