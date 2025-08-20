
from typing import Optional, Dict
from comet_ml import Experiment
import logging

logger = logging.getLogger(__name__)

class CometLogger:
    """Log metrics and experiments to Comet ML."""
    
    def __init__(self, api_key: Optional[str], project_name: Optional[str]):
        self.api_key = api_key
        self.project_name = project_name
        self.experiment = None
    
    def start_experiment(self, experiment_name: str) -> bool:
        """Start a Comet ML experiment."""
        if not self.api_key or not self.project_name:
            logger.warning("Comet ML API key or project name not provided, skipping logging")
            return False
        
        try:
            self.experiment = Experiment(
                api_key=self.api_key,
                project_name=self.project_name,
                experiment_name=experiment_name
            )
            logger.info(f"Started Comet ML experiment: {experiment_name}")
            return True
        except Exception as e:
            logger.error(f"Error starting Comet ML experiment: {str(e)}")
            return False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to Comet ML."""
        if not self.experiment:
            logger.warning("Comet ML experiment not started, skipping logging")
            return
        
        try:
            self.experiment.log_metrics(metrics, step=step)
            logger.info(f"Logged metrics to Comet ML: {metrics}")
        except Exception as e:
            logger.error(f"Error logging metrics to Comet ML: {str(e)}")
            raise
    
    def end_experiment(self) -> None:
        """End the Comet ML experiment."""
        if self.experiment:
            try:
                self.experiment.end()
                logger.info("Ended Comet ML experiment")
            except Exception as e:
                logger.error(f"Error ending Comet ML experiment: {str(e)}")