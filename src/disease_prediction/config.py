"""Paths and shared constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "disease_train.csv"
TEST_PATH = DATA_DIR / "disease_test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "final_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "test_predictions.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TARGET = "diagnosis"
ID_COLUMN = "patient_id"
FEATURES = [f"feature_{i}" for i in range(1, 11)]

RANDOM_SEED = 42
VALIDATION_SIZE = 0.2
CV_FOLDS = 5
