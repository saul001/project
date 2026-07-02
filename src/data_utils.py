"""
Data loading, cleaning, feature engineering and sequence-building utilities.
These functions are shared by train.py, forecast.py and app.py so that the
exact same preprocessing is applied everywhere (training, evaluation, and
live 1-week-ahead forecasting on new data).
"""
import numpy as np
import pandas as pd
from datetime import timedelta

from src import config


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw_csv(csv_path):
    """Load the raw multi-company CSV. Expected columns:
    symbol, date, open, high, low, close, volume
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    df["date"] = pd.to_datetime(df["date"])

    # Numeric OHLCV columns sometimes arrive as strings (e.g. exported from
    # Excel with thousands separators like "99,476.00", stray whitespace, or
    # currency symbols). Strip that formatting before casting to float so
    # downstream .astype("float64") calls don't blow up.
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("Rs.", "", regex=False)
                .str.replace("Rs", "", regex=False)
                .str.strip()
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_rows = df[numeric_cols].isnull().any(axis=1).sum()
    if bad_rows:
        print(f"WARNING: {bad_rows} row(s) had non-numeric OHLCV values that "
              f"could not be parsed and were set to NaN (will be filled by "
              f"clean_series()).")

    return df


def split_by_company(raw_df, companies=None):
    companies = companies or config.COMPANIES
    company_dfs = {}
    for sym in companies:
        sub = raw_df[raw_df["symbol"] == sym].sort_values("date").reset_index(drop=True)
        if len(sub) == 0:
            continue
        company_dfs[sym] = sub
    return company_dfs


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_series(df):
    """Drop duplicate dates, forward/back-fill gaps, winsorize outliers on OHLC."""
    d = df.copy()
    d = d.drop_duplicates(subset="date")
    d = d.sort_values("date").reset_index(drop=True)

    ohlcv = ["open", "high", "low", "close", "volume"]
    missing_before = int(d[ohlcv].isnull().sum().sum())
    d[ohlcv] = d[ohlcv].ffill().bfill()

    q1, q3 = d["close"].quantile(0.25), d["close"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    outliers = int(((d["close"] < lower) | (d["close"] > upper)).sum())
    for col in ["close", "open", "high", "low"]:
        d[col] = d[col].clip(lower, upper)

    return d, missing_before, outliers


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------
def add_technical_indicators(df):
    """Adds SMA/EMA/RSI/MACD/Bollinger Bands/returns/volatility columns.
    Requires at least ~26-30 rows of history to produce non-NaN rows at the
    tail (MACD needs the longest warm-up window of the indicators used).
    """
    d = df.copy()

    d["SMA_10"] = d["close"].rolling(window=10).mean()
    d["SMA_20"] = d["close"].rolling(window=20).mean()
    d["EMA_10"] = d["close"].ewm(span=10, adjust=False).mean()
    d["EMA_20"] = d["close"].ewm(span=20, adjust=False).mean()

    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    d["RSI_14"] = 100 - (100 / (1 + rs))

    ema_12 = d["close"].ewm(span=12, adjust=False).mean()
    ema_26 = d["close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema_12 - ema_26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()

    sma20 = d["close"].rolling(window=20).mean()
    std20 = d["close"].rolling(window=20).std()
    d["BB_upper"] = sma20 + 2 * std20
    d["BB_lower"] = sma20 - 2 * std20

    d["Daily_Return"] = d["close"].pct_change()
    d["Volatility_10"] = d["Daily_Return"].rolling(window=10).std()

    return d


def build_indicator_dataset(clean_df):
    ind = add_technical_indicators(clean_df)
    ind = ind.dropna().reset_index(drop=True)
    return ind


# ---------------------------------------------------------------------------
# Sequence building for supervised LSTM/GRU training
# ---------------------------------------------------------------------------
def create_sequences(data, window_size, target_idx):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, :])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)


def prepare_company_data(df, feature_cols=None, target_col=None,
                          window_size=None, train_split=None):
    from sklearn.preprocessing import MinMaxScaler

    feature_cols = feature_cols or config.FEATURE_COLS
    target_col = target_col or config.TARGET_COL
    window_size = window_size or config.WINDOW_SIZE
    train_split = train_split or config.TRAIN_SPLIT

    values = df[feature_cols].values.astype("float64")
    target_idx = feature_cols.index(target_col)

    split_idx = int(len(values) * train_split)
    train_raw, test_raw = values[:split_idx], values[split_idx:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    # Prepend the tail of train data to test so the first test windows have
    # enough look-back context.
    test_scaled_full = np.vstack([train_scaled[-window_size:], test_scaled])

    X_train, y_train = create_sequences(train_scaled, window_size, target_idx)
    X_test, y_test = create_sequences(test_scaled_full, window_size, target_idx)

    dates_test = df["date"].values[split_idx:]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler, "target_idx": target_idx,
        "n_features": len(feature_cols),
        "dates_test": dates_test,
        "feature_cols": feature_cols,
    }


def inverse_transform_target(scaled_target, scaler, n_features, target_idx):
    scaled_target = np.asarray(scaled_target).reshape(-1)
    dummy = np.zeros((len(scaled_target), n_features))
    dummy[:, target_idx] = scaled_target
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_idx]


# ---------------------------------------------------------------------------
# NEPSE calendar helper (NEPSE trades Sunday-Thursday)
# ---------------------------------------------------------------------------
def next_nepse_trading_day(d):
    nd = pd.Timestamp(d) + timedelta(days=1)
    while nd.weekday() in config.NEPSE_WEEKEND:
        nd += timedelta(days=1)
    return nd


def full_preprocess_pipeline(raw_df, symbol):
    """Convenience wrapper: raw multi-company df -> cleaned, indicator-added
    single-company df, ready for prepare_company_data()."""
    sub = raw_df[raw_df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
    cleaned, _, _ = clean_series(sub)
    ind = build_indicator_dataset(cleaned)
    return ind
