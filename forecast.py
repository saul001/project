"""
1-week-ahead (multi-step, recursive) forecasting.

Given a trained model + scaler for a company, and the most recent raw OHLCV
history (either the bundled processed data or a freshly-uploaded CSV of new
data), this recursively predicts the next N NEPSE trading days of closing
price:

  1. Recompute technical indicators on the raw OHLCV history available so far
  2. Take the last WINDOW_SIZE rows of engineered features, scale them
  3. Predict the next day's close
  4. Append a synthetic next-day OHLCV row (open = prev close, high/low a
     small band around it, volume = recent rolling average) so indicators
     can be recomputed for the following step
  5. Repeat for N_DAYS, skipping Friday/Saturday (NEPSE is closed)

This is a standard, clearly-documented simplification for the non-target
OHLCV fields — only `close` is actually predicted by the model; other
fields are carried forward with the same simple assumption used by most
academic recursive-forecasting setups.
"""
import numpy as np
import pandas as pd

from src import config, data_utils


def forecast_next_days(model, scaler, raw_ohlcv_df, feature_cols=None,
                        target_col=None, window_size=None, n_days=None):
    """
    raw_ohlcv_df: DataFrame with at least ['date','open','high','low','close','volume']
                  columns, RAW (indicators not yet computed), sorted ascending
                  by date, with enough history to compute indicators + a full
                  look-back window (>= window_size + ~30 rows recommended).
    """
    feature_cols = feature_cols or config.FEATURE_COLS
    target_col = target_col or config.TARGET_COL
    window_size = window_size or config.WINDOW_SIZE
    n_days = n_days or config.FORECAST_HORIZON_DAYS
    target_idx = feature_cols.index(target_col)

    df = raw_ohlcv_df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    predictions, dates, lower_band, upper_band = [], [], [], []
    last_date = df["date"].iloc[-1]

    # rough recent volatility, used only to draw an illustrative uncertainty
    # band around each forecast point (NOT a statistical confidence interval)
    recent_returns = df["close"].pct_change().dropna().tail(30)
    daily_vol = float(recent_returns.std()) if len(recent_returns) > 5 else 0.02

    for step in range(1, n_days + 1):
        ind = data_utils.add_technical_indicators(df)
        ind = ind.dropna().reset_index(drop=True)
        if len(ind) < window_size:
            raise ValueError(
                f"Not enough history to forecast: need at least "
                f"{window_size} rows after indicator warm-up, have {len(ind)}. "
                f"Provide more historical rows."
            )

        window = ind[feature_cols].values[-window_size:].astype("float64")
        window_scaled = scaler.transform(window)
        X = window_scaled.reshape(1, window_size, len(feature_cols))

        pred_scaled = model.predict(X, verbose=0).flatten()
        pred_close = data_utils.inverse_transform_target(
            pred_scaled, scaler, len(feature_cols), target_idx)[0]

        next_date = data_utils.next_nepse_trading_day(last_date)
        predictions.append(float(pred_close))
        dates.append(next_date)
        # illustrative +/- band that widens with the forecast horizon
        band_width = pred_close * daily_vol * np.sqrt(step)
        lower_band.append(float(pred_close - band_width))
        upper_band.append(float(pred_close + band_width))
        last_date = next_date

        prev_close = float(df["close"].iloc[-1])
        recent_vol = float(df["volume"].tail(10).mean())
        new_row = {
            "date": next_date,
            "open": prev_close,
            "close": float(pred_close),
            "high": max(prev_close, pred_close) * 1.001,
            "low": min(prev_close, pred_close) * 0.999,
            "volume": recent_vol,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame({
        "date": dates,
        "predicted_close": predictions,
        "lower_band": lower_band,
        "upper_band": upper_band,
        "day_ahead": range(1, n_days + 1),
    })


def forecast_ensemble(models_dict, scaler, raw_ohlcv_df, **kwargs):
    """Run forecast_next_days() for each model in models_dict = {'LSTM': model, 'GRU': model}
    and return a dict of {model_type: forecast_df}."""
    return {
        model_type: forecast_next_days(model, scaler, raw_ohlcv_df, **kwargs)
        for model_type, model in models_dict.items()
    }


if __name__ == "__main__":
    import argparse
    import pickle
    import os

    parser = argparse.ArgumentParser(description="Forecast the next N trading days for a company")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--model-type", type=str, choices=["LSTM", "GRU"], default="LSTM")
    parser.add_argument("--csv", type=str, default=None,
                         help="Optional path to new raw OHLCV data (symbol,date,open,high,low,close,volume). "
                              "If omitted, uses the bundled processed history for --symbol.")
    parser.add_argument("--days", type=int, default=config.FORECAST_HORIZON_DAYS)
    args = parser.parse_args()

    from tensorflow.keras.models import load_model

    model_path = os.path.join(config.MODELS_DIR, f"{args.symbol}_{args.model_type}.keras")
    scaler_path = os.path.join(config.MODELS_DIR, f"{args.symbol}_scaler.pkl")
    model = load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    if args.csv:
        raw = data_utils.load_raw_csv(args.csv)
        raw = raw[raw["symbol"] == args.symbol]
    else:
        processed_path = os.path.join(config.DATA_DIR, f"{args.symbol}_processed.csv")
        raw = pd.read_csv(processed_path)

    result = forecast_next_days(model, scaler, raw, n_days=args.days)
    print(result.to_string(index=False))

    out_path = os.path.join(config.FORECASTS_DIR, f"{args.symbol}_{args.model_type}_forecast.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved forecast to: {out_path}")
