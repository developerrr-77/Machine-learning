import os

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

settings = Settings()