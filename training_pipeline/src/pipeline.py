
import pandas as pd
from typing import Dict, List
from pathlib import Path
import logging
from load_data import DataLoader
from model_trainer import ModelTrainer
from hopsworks_manager import HopsworksManager
from config import TrainingPipelineConfig
from ensemble import Ensemble
from comet_ml import Experiment

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Training pipeline for demand forecasting."""
    
    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        self.loader = DataLoader(config.feature_path)
        self.trainer = ModelTrainer(n_splits=config.n_splits, test_size=config.test_size)
        self.hopsworks = HopsworksManager(config.hopsworks_api_key, config.hopsworks_project_name)
        self.experiment = Experiment(api_key=config.comet_api_key, project_name=config.comet_project_name)
        self.experiment.set_name("demand_forecasting")
        logger.info("Started Comet ML experiment: demand_forecasting")
        self.ensemble = Ensemble()
    
    def run(self) -> Dict[str, Dict[str, float]]:
        """Run the training pipeline for all SKUs."""
        results = {}
        df = self.loader.load_features()
        logger.info(f"Loaded features from {self.config.feature_path}, shape: {df.shape}")
        
        # Check if 'SkuName' exists in the DataFrame
        if 'SkuName' not in df.columns:
            logger.error(f"'SkuName' column not found in DataFrame. Available columns: {df.columns.tolist()}")
            raise KeyError(f"'SkuName' column not found in DataFrame. Available columns: {df.columns.tolist()}")
        
        if not self.hopsworks.connect():
            logger.warning("Hopsworks not connected, models will be saved locally only")
        
        for sku in df['SkuName'].unique():
            sku = str(sku)
            logger.info(f"Training models for SKU {sku}")
            sku_df = df[df['SkuName'] == sku]
            
            if len(sku_df) < self.config.min_data_points:
                logger.warning(f"Skipping SKU {sku} with {len(sku_df)} rows, less than min_data_points={self.config.min_data_points}")
                continue
            
            # Train Prophet
            prophet_model, prophet_metrics = self.trainer.train_prophet(sku_df)
            if prophet_model is not None and prophet_metrics is not None:
                self.trainer.save_model(prophet_model, "prophet", sku, self.config.model_dir)
                self.experiment.log_metrics(
                    {f"prophet_mae": prophet_metrics['mae'], f"prophet_rmse": prophet_metrics['rmse']},
                    step=0
                )
                logger.info(f"Logged metrics to Comet ML: {prophet_metrics}")
                if self.hopsworks.connect():
                    self.hopsworks.save_model(prophet_model, f"{sku}_prophet", prophet_metrics)
            
            # Train XGBoost
            xgboost_model, xgboost_metrics = self.trainer.train_xgboost(sku_df)
            if xgboost_model is not None and xgboost_metrics is not None:
                self.trainer.save_model(xgboost_model, "xgboost", sku, self.config.model_dir)
                self.experiment.log_metrics(
                    {f"xgboost_mae": xgboost_metrics['mae'], f"xgboost_rmse": xgboost_metrics['rmse']},
                    step=0
                )
                logger.info(f"Logged metrics to Comet ML: {xgboost_metrics}")
                if self.hopsworks.connect():
                    self.hopsworks.save_model(xgboost_model, f"{sku}_xgboost", xgboost_metrics)
            
            # Ensemble evaluation
            if prophet_model is not None and xgboost_model is not None:
                ensemble_metrics = self.ensemble.evaluate(prophet_model, xgboost_model, sku_df)
                results[sku] = {
                    'prophet_metrics': prophet_metrics,
                    'xgboost_metrics': xgboost_metrics,
                    'ensemble_metrics': ensemble_metrics
                }
                self.experiment.log_metrics(
                    {f"ensemble_mae": ensemble_metrics['mae'], f"ensemble_rmse": ensemble_metrics['rmse']},
                    step=0
                )
                logger.info(f"Ensemble metrics: {sku} {ensemble_metrics}")
                logger.info(f"Logged metrics to Comet ML: {ensemble_metrics}")
            
        return results
