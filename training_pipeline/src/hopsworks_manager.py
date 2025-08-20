
from typing import Optional, Dict
import hopsworks
from pathlib import Path
import logging
import joblib
import tempfile

logger = logging.getLogger(__name__)

class HopsworksManager:
    """Manage Hopsworks model registry integration."""
    
    def __init__(self, api_key: Optional[str], project_name: Optional[str]):
        self.api_key = api_key
        self.project_name = project_name
        self.project = None
        self.model_registry = None
    
    def connect(self) -> bool:
        """Connect to Hopsworks project."""
        if not self.api_key or not self.project_name:
            logger.warning("Hopsworks API key or project name not provided, skipping connection")
            return False
        
        try:
            self.project = hopsworks.login(api_key_value=self.api_key, project_name=self.project_name)
            self.model_registry = self.project.get_model_registry()
            logger.info(f"Connected to Hopsworks project: {self.project_name}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Hopsworks: {str(e)}")
            return False
    
    def save_model(self, model: object, model_name: str, metrics: Dict[str, float]) -> None:
        """Save model to Hopsworks model registry."""
        if not self.model_registry:
            logger.warning("Hopsworks not connected, skipping model save")
            return
        
        try:
            # Save model to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
                joblib.dump(model, tmp.name)
                tmp_path = tmp.name
                
                # Create model in Hopsworks model registry
                model_entry = self.model_registry.model(
                    name=model_name,
                    metrics=metrics,
                    version=None  # Auto-increment version
                )
                
                # Upload model file
                model_entry.save(model_path=tmp_path)
                logger.info(f"Saved model {model_name} to Hopsworks model registry")
                
                # Clean up temporary file
                Path(tmp_path).unlink()
        
        except Exception as e:
            logger.error(f"Error saving model {model_name} to Hopsworks: {str(e)}")
            raise
