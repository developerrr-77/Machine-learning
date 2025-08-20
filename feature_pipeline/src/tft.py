# src/tft_training.py
# Temporal Fusion Transformer training for multi-item demand forecasting.
# Production-focused with all fixes for PyTorch Lightning 2.0+ compatibility.

import os
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

# Import comet_ml before torch to avoid partial logging issues
from comet_ml import API as CometAPI
from comet_ml import Experiment

import pandas as pd
import torch
import pytorch_lightning as pl
from dotenv import load_dotenv

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger, CSVLogger

# ----------------------------
# Environment and constants
# ----------------------------
load_dotenv()

# Configuration
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "")
FG_NAME = os.getenv("FG_NAME", "demand_records")
FG_VERSION = int(os.getenv("FG_VERSION", "1"))

TARGET = os.getenv("TARGET_COL", "quantity")
DATE_COL = os.getenv("DATE_COL", "order_date")
ITEM_COL = os.getenv("ITEM_COL", "product_id")
STATIC_CAT_COLS = [c.strip() for c in os.getenv("STATIC_CAT_COLS", "supplier").split(",") if c.strip()]

ENCODER_LENGTH = int(os.getenv("ENCODER_LENGTH", "30"))
PREDICTION_LENGTH = int(os.getenv("PREDICTION_LENGTH", "14"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "30"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", "32"))
ATTN_HEADS = int(os.getenv("ATTENTION_HEADS", "4"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))
HIDDEN_CONT_SIZE = int(os.getenv("HIDDEN_CONT_SIZE", "16"))

COMET_API_KEY = os.getenv("COMET_API_KEY", "")
COMET_WORKSPACE = os.getenv("COMET_WORKSPACE", "")
COMET_PROJECT = os.getenv("COMET_PROJECT", "tft-forecasting")
COMET_EXPERIMENT_NAME = os.getenv("COMET_EXPERIMENT_NAME", "tft_demand_forecast")
COMET_REGISTRY_NAME = os.getenv("COMET_REGISTRY_NAME", "tft_demand_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
warnings.filterwarnings("ignore", message=".*save_hyperparameters.*")
pl.seed_everything(42, workers=True)
torch.set_num_threads(max(1, NUM_WORKERS))

# ----------------------------
# TFT Lightning Wrapper
# ----------------------------
# Update the TFTLightningWrapper class
class TFTLightningWrapper(pl.LightningModule):
    """Wrapper to properly integrate TFT with PyTorch Lightning 2.0+"""
    def __init__(self, tft_model: TemporalFusionTransformer):
        super().__init__()
        self.save_hyperparameters(ignore=["tft_model"])
        self.tft_model = tft_model
        self._current_stage = None
        
        # Add missing attributes required by PyTorch Forecasting
        self.tft_model.predicting = property(lambda self: self._current_stage == "predict")
        self.tft_model.current_stage = property(lambda self: self._current_stage)
        self.tft_model.trainer = property(lambda self: self.trainer)

    @property
    def current_stage(self):
        return self._current_stage

    @current_stage.setter
    def current_stage(self, value):
        self._current_stage = value

    def forward(self, x):
        return self.tft_model(x)

    def training_step(self, batch, batch_idx):
        self.current_stage = "train"
        return self.tft_model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.current_stage = "validate"
        return self.tft_model.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.current_stage = "predict"
        return self.tft_model.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def configure_optimizers(self):
        return self.tft_model.configure_optimizers()
# ----------------------------
# Data Loading
# ----------------------------
def load_data(filename: str = "transformed_data.xlsx") -> pd.DataFrame:
    """Load data from Hopsworks or local file with daily reindexing."""
    try:
        import hopsworks
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
        fs = project.get_feature_store()
        df = fs.get_feature_group(FG_NAME, version=FG_VERSION).read()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        logging.info("Data loaded from Hopsworks Feature Store")
    except Exception as e:
        logging.warning(f"Hopsworks unavailable, using local file. Error: {e}")
        file_path = Path(__file__).parent.parent / "data" / "transformed" / filename
        df = pd.read_excel(file_path) if file_path.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(file_path)

    # Validate and preprocess
    required = {DATE_COL, ITEM_COL, TARGET}
    if missing := required - set(df.columns):
        raise KeyError(f"Missing columns: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dropna()
    df = df.sort_values([ITEM_COL, DATE_COL])
    
    # Daily reindexing per series
    filled = []
    for pid, g in df.groupby(ITEM_COL, sort=False):
        g = g.set_index(DATE_COL).sort_index()
        all_days = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(all_days)
        g[ITEM_COL] = pid
        for c in STATIC_CAT_COLS:
            if c in g.columns:
                g[c] = g[c].ffill().bfill()
        g[TARGET] = pd.to_numeric(g[TARGET], errors="coerce").fillna(0).clip(lower=0)
        filled.append(g.reset_index().rename(columns={"index": DATE_COL}))

    if not filled:
        raise ValueError("No valid series after processing")
    
    df = pd.concat(filled, ignore_index=True)
    df["time_idx"] = (df[DATE_COL] - df[DATE_COL].min()).dt.days.astype(int)
    df["dayofweek"] = df[DATE_COL].dt.dayofweek.astype(int)
    df["month"] = df[DATE_COL].dt.month.astype(int)
    
    return df

# ----------------------------
# Dataset Preparation
# ----------------------------
def build_datasets(df: pd.DataFrame) -> Tuple[TimeSeriesDataSet, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create training and validation datasets with proper time series splitting."""
    min_len = ENCODER_LENGTH + PREDICTION_LENGTH
    valid_items = df.groupby(ITEM_COL).size()[lambda x: x >= min_len].index
    df = df[df[ITEM_COL].isin(valid_items)].copy()
    cutoff = df[DATE_COL].max() - pd.Timedelta(days=PREDICTION_LENGTH)

    static_cats = [ITEM_COL] + [c for c in STATIC_CAT_COLS if c in df.columns]
    
    training = TimeSeriesDataSet(
        df[df[DATE_COL] <= cutoff],
        time_idx="time_idx",
        target=TARGET,
        group_ids=[ITEM_COL],
        min_encoder_length=ENCODER_LENGTH // 2,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=PREDICTION_LENGTH,
        static_categoricals=static_cats,
        time_varying_known_reals=["time_idx", "dayofweek", "month"],
        time_varying_unknown_reals=[TARGET],
        categorical_encoders={col: NaNLabelEncoder(add_nan=True) for col in static_cats},
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    
    train_loader = training.to_dataloader(
        train=True, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        persistent_workers=False
    )
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=False
    )
    
    return training, train_loader, val_loader

# ----------------------------
# Model Training
# ----------------------------
def build_tft(training: TimeSeriesDataSet) -> TFTLightningWrapper:
    """Build and configure the TFT model with wrapper."""
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTN_HEADS,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONT_SIZE,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
        output_size=7,
    )
    wrapper = TFTLightningWrapper(tft)
    wrapper.current_stage = None  # Initialize stage
    return wrapper

def train_model(
    training: TimeSeriesDataSet,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    logger: Optional[pl.loggers.Logger]
) -> Tuple[TemporalFusionTransformer, str]:
    """Train the model with proper trainer integration."""
    tft_wrapper = build_tft(training)
    
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=str(CHECKPOINT_DIR),
            filename="tft-best",
            save_top_k=1,
            mode="min",
        ),
        EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ]
    
    if logger is None:
        logger = CSVLogger(save_dir=str(ARTIFACTS_DIR), name="tft_runs")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=0.1,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(tft_wrapper, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    if not (best_path := callbacks[0].best_model_path):
        raise RuntimeError("Training failed to produce a valid checkpoint")
    
    best_wrapper = TFTLightningWrapper.load_from_checkpoint(best_path)
    return best_wrapper.tft_model, best_path

# ----------------------------
# Evaluation & Registry
# ----------------------------
def evaluate(
    model: TemporalFusionTransformer,
    val_loader: torch.utils.data.DataLoader,
    logger: Optional[pl.loggers.Logger]
) -> Dict[str, float]:
    """Evaluate model performance and log results."""
    wrapper = TFTLightningWrapper(model)
    wrapper.eval()
    
    with torch.no_grad():
        preds = wrapper.predict(val_loader).cpu().float().squeeze()
        actuals = torch.cat([y[0] for _, y in iter(val_loader)]).cpu().float().squeeze()

    preds = preds[:len(actuals)]
    metrics = {
        "rmse": torch.sqrt(torch.mean((preds - actuals) ** 2)).item(),
        "mae": torch.mean(torch.abs(preds - actuals)).item()
    }
    
    logging.info(f"Validation RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")

    if isinstance(logger, CometLogger):
        logger.log_metrics(metrics)
        metrics_path = ARTIFACTS_DIR / "val_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        logger.experiment.log_asset(str(metrics_path))

    return metrics

def register_model(best_ckpt_path: str, metrics: Dict[str, float]) -> None:
    """Register model in Comet.ml registry if configured."""
    if not COMET_API_KEY or not COMET_WORKSPACE:
        logging.warning("Skipping Comet registry - missing credentials")
        return
    
    metadata = {
        "framework": "pytorch-forecasting",
        "model": "TemporalFusionTransformer",
        "encoder_length": ENCODER_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "metrics": metrics,
        "description": "Production TFT for demand forecasting",
    }
    
    CometAPI(COMET_API_KEY).upload_registry_model(
        workspace=COMET_WORKSPACE,
        registry_name=COMET_REGISTRY_NAME,
        version=MODEL_VERSION,
        model=best_ckpt_path,
        metadata=metadata,
    )
    logging.info(f"Registered model {COMET_REGISTRY_NAME}:{MODEL_VERSION}")

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # Version checks
    assert pl.__version__ >= "2.0.0", "Requires PyTorch Lightning 2.0+"
    assert hasattr(TemporalFusionTransformer, "from_dataset"), "Requires pytorch-forecasting 1.0+"
    assert torch.__version__ >= "2.0.0", "Requires PyTorch 2.0+"

    # Load and prepare data
    df = load_data()
    logging.info(f"Data loaded: {len(df)} rows, {df[ITEM_COL].nunique()} items")

    # Initialize logging
    comet_logger = None
    if COMET_API_KEY and COMET_WORKSPACE:
        comet_logger = CometLogger(
            api_key=COMET_API_KEY,
            workspace=COMET_WORKSPACE,
            project_name=COMET_PROJECT,
            experiment_name=COMET_EXPERIMENT_NAME,
            save_dir=str(ARTIFACTS_DIR),
        )

    # Build datasets and train
    training, train_loader, val_loader = build_datasets(df)
    best_model, best_ckpt_path = train_model(training, train_loader, val_loader, comet_logger)

    # Evaluate and register
    metrics = evaluate(best_model, val_loader, comet_logger)
    register_model(best_ckpt_path, metrics)

    # Clean up
    if isinstance(comet_logger, CometLogger):
        comet_logger.experiment.end()