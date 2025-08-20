
from config import TrainingPipelineConfig
from pipeline import TrainingPipeline
import logging

def main():
    """Run the training pipeline."""
    try:
        config = TrainingPipelineConfig()
        pipeline = TrainingPipeline(config)
        results = pipeline.run()
        if results:
            logging.info("Training pipeline completed successfully")
            logging.info(f"Training results: {results}")
        else:
            logging.error("Training pipeline failed")
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
