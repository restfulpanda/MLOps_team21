from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
DATASET_PATH = DATA_DIR / "winequality-red.csv"

PROC_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR/ "model.pkl"

REPORT_DIR = ROOT_DIR / "reports"

DATASET_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
