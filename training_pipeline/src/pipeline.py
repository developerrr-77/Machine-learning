import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
from config import TrainingPipelineConfig
from model_trainer import ModelTrainer
from ensemble import EnsemblePredictor
from comet_ml import Experiment  # Import comet_ml first
import os

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Training pipeline for demand forecasting."""
    
    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        # Initialize Comet.ml Experiment
        self.experiment = Experiment(
            api_key=config.comet_api_key,
            project_name=config.comet_project_name,
            display_summary_level=0
        )
        self.experiment.set_name("demand_forecasting")
        # Initialize ModelTrainer after experiment
        self.trainer = ModelTrainer(n_splits=config.n_splits, test_size=config.test_size, experiment=self.experiment)
    
    def detect_outliers(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Detect and separate outliers using IQR method per SKU."""
        try:
            def find_outliers(group):
                if len(group) < 3:  # Skip if too few data points
                    logger.warning(f"Skipping outlier detection for SKU {group.name}: {len(group)} data points < 3")
                    return pd.Series(False, index=group.index)
                Q1 = group['NoOfPItems'].quantile(0.25)
                Q3 = group['NoOfPItems'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return group['NoOfPItems'].between(lower_bound, upper_bound)
            
            # Apply outlier detection per SKU
            is_not_outlier = df.groupby('SkuName').apply(find_outliers).reset_index(level=0, drop=True)
            cleaned_df = df[is_not_outlier].copy()
            outlier_df = df[~is_not_outlier].copy()
            
            logger.info(f"Cleaned dataset: {len(cleaned_df)} rows, Outlier dataset: {len(outlier_df)} rows")
            self.experiment.log_parameter("cleaned_rows", len(cleaned_df))
            self.experiment.log_parameter("outlier_rows", len(outlier_df))
            return cleaned_df, outlier_df
        
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            raise
    
    def save_dataset(self, df: pd.DataFrame, path: str) -> None:
        """Save dataset to parquet file."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            logger.info(f"Saved dataset to {path}")
        except Exception as e:
            logger.error(f"Error saving dataset to {path}: {str(e)}")
            raise
    
    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing NoOfPItems values with forward fill or mean."""
        try:
            df = df.copy()
            # Forward fill for time-series continuity
            df['NoOfPItems'] = df.groupby('SkuName')['NoOfPItems'].ffill()
            # Fill remaining NaNs with mean per SKU
            df['NoOfPItems'] = df.groupby('SkuName')['NoOfPItems'].transform(lambda x: x.fillna(x.mean()))
            # Log remaining NaNs
            remaining_nans = df['NoOfPItems'].isna().sum()
            if remaining_nans > 0:
                logger.warning(f"Remaining NaN values after imputation: {remaining_nans}")
            return df
        except Exception as e:
            logger.error(f"Error imputing missing values: {str(e)}")
            raise
    
    def run(self) -> Dict[str, Dict[str, float]]:
        """Run the training pipeline for all SKUs."""
        try:
            results = {}
            df = pd.read_parquet(self.config.feature_path)
            logger.info(f"Loaded data: {df.shape}")
            self.experiment.log_parameter("input_rows", df.shape[0])
            
            # Impute missing values
            df = self.impute_missing_values(df)
            logger.info(f"Data after imputation: {df.shape}")
            
            # Detect and separate outliers
            cleaned_df, outlier_df = self.detect_outliers(df)
            self.save_dataset(cleaned_df, '/workspaces/Machine-learning/data/transformed/cleaned_features.parquet')
            self.save_dataset(outlier_df, '/workspaces/Machine-learning/data/transformed/outlier_features.parquet')
            
            # Process both datasets
            for dataset_name, dataset, model_dir in [
                ('cleaned', cleaned_df, f"{self.config.model_dir}/cleaned"),
                ('outliers', outlier_df, f"{self.config.model_dir}/outliers")
            ]:
                logger.info(f"Processing {dataset_name} dataset with {len(dataset)} rows")
                self.experiment.log_parameter(f"{dataset_name}_rows", len(dataset))
                
                for i, sku in enumerate(dataset['SkuName'].unique()):
                    logger.info(f"Training models for SKU {sku} in {dataset_name} dataset")
                    sku_df = dataset[dataset['SkuName'] == sku]
                    
                    # Log data points for debugging
                    logger.info(f"SKU {sku} has {len(sku_df)} data points, non-NaN NoOfPItems: {sku_df['NoOfPItems'].notna().sum()}")
                    
                    if len(sku_df) < self.config.min_data_points:
                        logger.warning(f"Skipping SKU {sku} in {dataset_name}: {len(sku_df)} data points < {self.config.min_data_points}")
                        self.experiment.log_parameter(f"skipped_sku_{dataset_name}_{i}", sku)
                        continue
                    
                    # Train Prophet model
                    prophet_model, prophet_metrics = self.trainer.train_prophet(sku_df)
                    if prophet_model:
                        self.trainer.save_model(prophet_model, 'prophet', sku, model_dir)
                        self.experiment.log_metrics({f"{dataset_name}_prophet_mae": prophet_metrics['mae'], f"{dataset_name}_prophet_rmse": prophet_metrics['rmse']}, step=i)
                        self.experiment.log_parameter(f"{dataset_name}_sku_{i}", sku)
                    
                    # Train XGBoost model
                    xgboost_model, xgboost_metrics = self.trainer.train_xgboost(sku_df)
                    if xgboost_model:
                        self.trainer.save_model(xgboost_model, 'xgboost', sku, model_dir)
                        self.experiment.log_metrics({f"{dataset_name}_xgboost_mae": xgboost_metrics['mae'], f"{dataset_name}_xgboost_rmse": xgboost_metrics['rmse']}, step=i)
                        self.experiment.log_parameter(f"{dataset_name}_sku_{i}", sku)
                    
                    # Ensemble predictions
                    if prophet_model or xgboost_model:
                        ensemble = EnsemblePredictor(prophet_model, xgboost_model)
                        ensemble_metrics = ensemble.evaluate(sku_df)
                        results[f"{dataset_name}_{sku}"] = ensemble_metrics
                        logger.info(f"Ensemble metrics for {sku} in {dataset_name}: {ensemble_metrics}")
                        self.experiment.log_metrics({f"{dataset_name}_ensemble_mae": ensemble_metrics['mae'], f"{dataset_name}_ensemble_rmse": ensemble_metrics['rmse']}, step=i)
                        self.experiment.log_parameter(f"{dataset_name}_sku_{i}", sku)
            
            return results
        
        except Exception as e:
            logger.error(f"Main execution failed: {str(e)}")
            self.experiment.log_other("error", str(e))
            raise
        finally:
            self.experiment.end()