"""
Central configuration for the NEPSE Manufacturing Stock Prediction project.
Edit this file to point at your own data / change hyperparameters — every
other script (train.py, forecast.py, app.py) reads from here so you only
need to change things in one place.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
FORECASTS_DIR = os.path.join(OUTPUTS_DIR, "forecasts")

for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, METRICS_DIR, PLOTS_DIR, FORECASTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Default dataset. Replace this file with your real NEPSE data, or pass a
# different --csv path to train.py. Required columns: symbol,date,open,high,low,close,volume
DEFAULT_CSV = os.path.join(DATA_DIR, "nepse_manufacturing.csv")

# ---------------------------------------------------------------------------
# Companies (5 manufacturing companies listed on NEPSE, as used in the
# reference notebook). Edit this list to match the symbols in your CSV.
# ---------------------------------------------------------------------------
COMPANIES = ["BPCL", "HDL", "UNL", "NHDL", "SHIVM"]

COMPANY_FULL_NAMES = {
    "BPCL": "Bottlers Nepal (Balaju) Ltd.",
    "HDL": "Himalayan Distillery Ltd.",
    "UNL": "Unilever Nepal Ltd.",
    "NHDL": "National Hydro / Manufacturing Ltd.",
    "SHIVM": "Shivam Cements Ltd.",
}

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "SMA_10", "SMA_20", "EMA_10", "EMA_20",
    "RSI_14", "MACD", "MACD_signal", "BB_upper", "BB_lower",
    "Daily_Return", "Volatility_10",
]
TARGET_COL = "close"
TARGET_IDX = FEATURE_COLS.index(TARGET_COL)

# ---------------------------------------------------------------------------
# Sequence / train-test split
# ---------------------------------------------------------------------------
WINDOW_SIZE = 60      # look-back window (trading days)
TRAIN_SPLIT = 0.80    # 80% train / 20% test, chronological (no shuffling)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10
VALIDATION_SPLIT = 0.10
LEARNING_RATE = 0.001

# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
FORECAST_HORIZON_DAYS = 7      # "1 week" of NEPSE trading days ahead
# NEPSE trades Sunday-Thursday. Friday(4) and Saturday(5) are closed
# (Python weekday(): Monday=0 ... Sunday=6)
NEPSE_WEEKEND = {4, 5}

MODEL_TYPES = ["LSTM", "GRU"]
