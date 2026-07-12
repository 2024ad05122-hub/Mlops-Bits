"""
Central configuration for the Heart Disease MLOps project.

Keeping feature lists, paths, and constants in one place makes the pipeline
reproducible and easy to personalise. If you change the feature split or
target definition, do it here so training and serving stay in sync.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "heart.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"          # full sklearn pipeline
METADATA_PATH = MODELS_DIR / "model_metadata.json"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature schema  (edit these to personalise your feature engineering)
# ---------------------------------------------------------------------------
TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Human-readable descriptions (used in the API docs / validation messages)
FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting ECG results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of the peak exercise ST segment (0-2)",
    "ca": "Number of major vessels colored by fluoroscopy (0-4)",
    "thal": "Thalassemia (0-3)",
}

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MLFLOW_EXPERIMENT_NAME = "heart-disease-classification"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"   # SQLite backend (recommended in MLflow 3.x)
# View the UI with:  mlflow ui --backend-store-uri sqlite:///mlflow.db
