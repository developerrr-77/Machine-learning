'''import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from models import BaseModel, XGBoostModel, ModelTrainer, ProphetModel
from settings import settings
import comet_ml
from comet_ml import Experiment
import joblib
import psutil

logging.basicConfig(level=logging.INFO, format=settings.LOG_FORMAT)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = None) -> Dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = self.robust_mape(y_true, y_pred, epsilon)
        return {'mae': mae, 'rmse': rmse, 'mape': mape}
    
    def robust_mape(self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = None) -> float:
        if epsilon is None:
            epsilon = max(10.0, np.mean(y_true) * 0.1)
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

class CombinedModel(BaseModel):
    def __init__(self, models_dict: Dict[str, BaseModel], feature_columns: List[str], weights: Dict[str, float] = None, model_name: str = "CombinedModel", max_pred: float = None):
        super().__init__(model_name)
        self.models_dict = models_dict
        self.feature_columns = feature_columns
        self.weights = weights if weights else {'prophet': 0.7, 'xgboost': 0.3}
        self.max_pred = max_pred
        self.is_fitted = all(model.is_fitted for model in models_dict.values())
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'CombinedModel':
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if 'product_id' not in X.columns:
            raise ValueError("Input DataFrame must have 'product_id' column")
        if 'date' not in X.columns:
            raise ValueError("Input DataFrame must have 'date' column")
        predictions = []
        for product_id, group in X.groupby('product_id'):
            model = self.models_dict.get(product_id)
            if model is None:
                mean_pred = group['demand'].mean() if 'demand' in group.columns else 0
                predictions.extend([mean_pred] * len(group))
                continue
            if isinstance(model, ProphetModel):
                # Pass DataFrame with 'date' column directly to ProphetModel
                pred = model.predict(group)
                pred = np.clip(pred, 0, self.max_pred) if self.max_pred else np.maximum(pred, 0)
                predictions.extend(pred)
            else:
                features = group[self.feature_columns].values
                pred = model.predict(pd.DataFrame(features, columns=self.feature_columns))
                pred = np.clip(pred, 0, self.max_pred) if self.max_pred else np.maximum(pred, 0)
                predictions.extend(pred)
        return np.array(predictions)
    
    def save_model(self, filepath: str) -> None:
        joblib.dump(self, filepath)
        logger.info(f"Combined model saved to {filepath}")

# Load environment variables
load_dotenv('/workspaces/Machine-learning/.env')

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage: RSS={mem_info.rss / 1024**2:.2f} MB, VMS={mem_info.vms / 1024**2:.2f} MB")

# Load raw data
try:
    df = pd.read_excel(settings.RAW_DATA_PATH)
    logger.info(f"Fetched data with shape: {df.shape}")
    logger.info(f"Raw data sample:\n{df.head()}")
except Exception as e:
    logger.error(f"Error loading data from {settings.RAW_DATA_PATH}: {e}")
    exit(1)

# Initialize components
preprocessor = DataPreprocessor()
feature_engineer = FeatureEngineer()
trainer = ModelTrainer()
evaluator = ModelEvaluator()

# Preprocess data with outlier detection and log-transformation
try:
    df_preprocessed = preprocessor.preprocess_demand_data(df, aggregate_by='weekly')
    Q1 = df_preprocessed['demand'].quantile(0.25)
    Q3 = df_preprocessed['demand'].quantile(0.75)
    IQR = Q3 - Q1
    df_preprocessed['demand'] = df_preprocessed['demand'].clip(upper=Q3 + 1.5 * IQR)
    df_preprocessed['demand'] = np.log1p(df_preprocessed['demand'] + 10)
    df_preprocessed['date'] = pd.to_datetime(df_preprocessed['date'], errors='coerce')
    df_preprocessed = df_preprocessed.dropna(subset=['date'])
except Exception as e:
    logger.error(f"Preprocessing failed: {e}")
    exit(1)

# Feature engineering with additional lags and product categories
try:
    df_features = feature_engineer.create_all_features(df_preprocessed, target_col='demand')
    df_features['demand_lag_3'] = df_features.groupby('product_id')['demand'].shift(3)
    df_features['demand_lag_6'] = df_features.groupby('product_id')['demand'].shift(6)
    df_features['demand_lag_12'] = df_features.groupby('product_id')['demand'].shift(12)
    df_features['holiday'] = df_features['date'].apply(lambda x: 1 if x.month == 12 and x.day == 25 else 0)
    medical_products = ['Syringes', 'Anti', 'Tube', 'Vial', 'Device', 'Inj', 'Cannula', 'Salin']
    df_features['is_medical'] = df_features['product_id'].apply(lambda x: 1 if any(p in x for p in medical_products) else 0)
    df_features = df_features.fillna(0)
except Exception as e:
    logger.error(f"Feature engineering failed: {e}")
    exit(1)

# Encode categorical features
df_features = preprocessor.encode_categorical_features(df_features)

# Select features
selected_features = feature_engineer.select_important_features(df_features, target_col='demand', max_features=18)
logger.info(f"Selected features: {selected_features}")

# Cluster products for sparse data
try:
    product_stats = df_features.groupby('product_id')['demand'].agg(['mean', 'std']).reset_index()
    kmeans = KMeans(n_clusters=5, random_state=42)
    product_stats['cluster'] = kmeans.fit_predict(product_stats[['mean', 'std']].fillna(0))
    product_clusters = dict(zip(product_stats['product_id'], product_stats['cluster']))
    df_features['cluster'] = df_features['product_id'].map(product_clusters)
except Exception as e:
    logger.error(f"Product clustering failed: {e}")
    df_features['cluster'] = 0

# Prepare data for modeling
product_data = preprocessor.prepare_for_modeling(df_features, target_col='demand')
if not product_data:
    logger.error("No products available for modeling.")
    exit(1)

# Filter products with insufficient data
min_data_points = 12
product_data = {pid: df for pid, df in product_data.items() if len(df) >= min_data_points}
logger.info(f"Filtered to {len(product_data)} products with at least {min_data_points} data points")

# Identify high-variance products
product_variances = {pid: df['demand'].var() for pid, df in product_data.items()}
variance_threshold = np.percentile(list(product_variances.values()), 95)
high_variance_products = [pid for pid, var in product_variances.items() if var >= variance_threshold]
low_variance_products = [pid for pid in product_data.keys() if pid not in high_variance_products]
logger.info(f"High-variance products ({len(high_variance_products)}): {high_variance_products}")
logger.info(f"Low-variance products ({len(low_variance_products)}): {low_variance_products}")

# Train/validation split
train_data = {}
val_data = {}
for product_id, product_df in product_data.items():
    test_size = max(2, int(len(product_df) * 0.05))
    train_df, val_df = preprocessor.create_train_test_split(product_df, test_size=test_size, time_based=True)
    if len(val_df) < 2:
        logger.warning(f"Skipping product_id {product_id}: insufficient validation data ({len(val_df)} records)")
        continue
    if (train_df['demand'] == 0).all():
        logger.warning(f"Skipping product_id {product_id}: all training demand is zero")
        continue
    train_data[product_id] = train_df
    val_data[product_id] = val_df
    logger.info(f"Product {product_id} - Train: {len(train_df)} records, Test: {len(val_df)} records")

# Aggregate statistics
train_quantity_mean = np.expm1(df_features['demand'] - 10).mean() if not df_features.empty else np.nan
train_quantity_std = np.expm1(df_features['demand'] - 10).std() if not df_features.empty else np.nan
val_quantity_mean = np.mean([np.expm1(val_df['demand'] - 10).mean() for val_df in val_data.values()]) if val_data else np.nan
max_pred = train_quantity_mean + 2 * train_quantity_std

# Initialize metrics storage
per_product_metrics = {}
high_mape_products = {}

# Train models
checkpoint_dir = os.path.join(settings.MODEL_DIR, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
models_dict = {}
checkpoint_file = os.path.join(checkpoint_dir, 'models_checkpoint.pkl')

try:
    log_memory_usage()
    for product_id in product_data.keys():
        if product_id in models_dict or product_id not in train_data:
            continue
        logger.info(f"Training model for product_id: {product_id}")
        X_train = train_data[product_id][selected_features + ['date']]  # Include 'date' for Prophet
        y_train = train_data[product_id]['demand']
        X_val = val_data[product_id][selected_features + ['date', 'product_id', 'demand']]  # Include 'date' and 'product_id'
        y_val = val_data[product_id]['demand']
        
        if len(X_train) < settings.MIN_HISTORICAL_DAYS:
            logger.warning(f"Skipping product_id {product_id}: insufficient training data ({len(X_train)} records)")
            continue
        
        model = ProphetModel()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        y_pred = np.expm1(y_pred - 10)
        y_val = np.expm1(y_val - 10)
        y_pred = np.clip(y_pred, 0, max_pred)
        metrics = evaluator.calculate_metrics(y_val, y_pred, epsilon=max(10.0, y_val.mean() * 0.1))
        
        if model is not None:
            models_dict[product_id] = model
        per_product_metrics[product_id] = metrics
        logger.info(f"Product {product_id} metrics: {metrics}")
        if metrics['mape'] > 100.0:
            high_mape_products[product_id] = metrics
        
        with open(checkpoint_file, 'wb') as f:
            joblib.dump(models_dict, f)
    log_memory_usage()
except Exception as e:
    logger.error(f"Training failed: {e}")
    with open(checkpoint_file, 'wb') as f:
        joblib.dump(models_dict, f)
    exit(1)

# Create and evaluate combined model
combined_model = CombinedModel(models_dict, selected_features, max_pred=max_pred)
all_val_predictions = []
all_val_actuals = []

for product_id in val_data.keys():
    X_val = val_data[product_id][selected_features + ['date', 'product_id', 'demand']]  # Include 'date'
    y_val = np.expm1(val_data[product_id]['demand'] - 10)
    if len(train_data[product_id]) < 3:
        y_pred = np.full_like(y_val, np.expm1(train_data[product_id]['demand'] - 10).mean() if len(train_data[product_id]) > 0 else 0)
    else:
        y_pred = combined_model.predict(X_val)
    metrics = evaluator.calculate_metrics(y_val, y_pred, epsilon=max(10.0, y_val.mean() * 0.1))
    per_product_metrics[product_id] = metrics
    logger.info(f"Validation metrics for {product_id}: {metrics}")
    if metrics['mape'] > 100.0:
        high_mape_products[product_id] = metrics
    elif metrics['mape'] <= 500.0 and not np.isnan(metrics['mape']):
        all_val_predictions.extend(y_pred)
        all_val_actuals.extend(y_val)

if not all_val_predictions:
    logger.error("No predictions generated.")
    exit(1)

all_val_predictions = np.array(all_val_predictions)
all_val_actuals = np.array(all_val_actuals)
metrics = evaluator.calculate_metrics(all_val_actuals, all_val_predictions, epsilon=max(10.0, all_val_actuals.mean() * 0.1))
logger.info(f"Combined Model MAE: {metrics['mae']:.4f}")
logger.info(f"Combined Model RMSE: {metrics['rmse']:.4f}")
logger.info(f"Combined Model MAPE: {metrics['mape']:.4f}%")
logger.info(f"High MAPE products (>100%): {len(high_mape_products)}")
logger.info(f"High MAPE product metrics: {high_mape_products}")

# Log to Comet ML
experiment = Experiment(
    api_key=settings.COMET_API_KEY,
    project_name=settings.COMET_PROJECT_NAME,
    workspace=settings.COMET_WORKSPACE
)

params = {
    'features': selected_features,
    'num_products_trained': len(models_dict),
    'high_variance_products': high_variance_products
}
experiment.log_parameters(params)
experiment.log_metrics(metrics)
experiment.log_metrics({'per_product_metrics': per_product_metrics, 'high_mape_products': high_mape_products})

data_stats = {
    'train_quantity_mean': train_quantity_mean,
    'train_quantity_std': train_quantity_std,
    'val_quantity_mean': val_quantity_mean,
    'val_quantity_std': np.mean([np.expm1(val_df['demand'] - 10).std() for val_df in val_data.values() if not np.isnan(np.expm1(val_df['demand'] - 10).std())]) if val_data else np.nan,
    'num_products': len(product_data)
}
experiment.log_metrics(data_stats)

model_path = os.path.join(settings.MODEL_DIR, 'combined_model.pkl')
combined_model.save_model(model_path)
experiment.log_model("combined_model", model_path)
experiment.end()'''


from data import ingest_data
from data_preprocessing import preprocess_data
from feature_engineering import create_features
from models import train_and_evaluate
from comet_ml import Experiment
import logging

logger = logging.getLogger(__name__)

def main():
    experiment = Experiment(project_name="inventory-predictions")

    df = ingest_data('/workspaces/Machine-learning/data/raw/SundasPoDetail.xlsx')
    df_preprocessed = preprocess_data()
    df_features = create_features(df_preprocessed)

    products = df_features['sku_name'].unique()
    for product in products:
        mae, rmse, mape = train_and_evaluate(df_features, product, experiment)
        logger.info(f"{product} - MAE: {mae}, RMSE: {rmse}, MAPE: {mape}")

    experiment.end()

if __name__ == "__main__":
    main()