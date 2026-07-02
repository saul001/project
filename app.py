"""
Streamlit dashboard for the NEPSE Manufacturing Stock Prediction project
(LSTM vs GRU).

Run with:
    streamlit run app.py

Features:
  - Company & model selector (LSTM / GRU / both)
  - Model performance dashboard: RMSE, MAE, MAPE, Accuracy(%), R2, Directional Accuracy
  - Directional (Up/Down) confusion matrix + classification report
  - Actual vs predicted price chart on the held-out test set, with residuals
  - 1-week-ahead (configurable) forecast for new/latest data, with an
    illustrative uncertainty band, and CSV download
  - Upload your own CSV of newer data to generate a fresh forecast without
    retraining
  - "Train models now" fallback if no trained models are found yet
"""
import os
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config, data_utils, model_utils, forecast

st.set_page_config(page_title="NEPSE Manufacturing Stock Prediction — LSTM vs GRU",
                    layout="wide", page_icon="📈")

# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_raw_csv(path_or_buffer):
    return data_utils.load_raw_csv(path_or_buffer)


@st.cache_resource(show_spinner=False)
def _load_keras_model(path):
    from tensorflow.keras.models import load_model
    return load_model(path)


@st.cache_resource(show_spinner=False)
def _load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def _load_metrics_summary():
    path = os.path.join(config.METRICS_DIR, "metrics_summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def _load_confusion_reports():
    path = os.path.join(config.METRICS_DIR, "confusion_reports.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def _load_predictions_csv(symbol, model_type):
    path = os.path.join(config.OUTPUTS_DIR, f"{symbol}_{model_type}_predictions.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        return df
    return None


def model_available(symbol, model_type):
    return os.path.exists(os.path.join(config.MODELS_DIR, f"{symbol}_{model_type}.keras"))


def scaler_available(symbol):
    return os.path.exists(os.path.join(config.MODELS_DIR, f"{symbol}_scaler.pkl"))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📈 Controls")

data_source = st.sidebar.radio(
    "Data source",
    ["Bundled dataset", "Upload new CSV"],
    help="Use the project's dataset, or upload newer OHLCV data "
         "(symbol,date,open,high,low,close,volume) to forecast on.",
)

if data_source == "Upload new CSV":
    uploaded = st.sidebar.file_uploader("Upload OHLCV CSV", type=["csv"])
    raw_df = _load_raw_csv(uploaded) if uploaded is not None else None
else:
    raw_df = _load_raw_csv(config.DEFAULT_CSV) if os.path.exists(config.DEFAULT_CSV) else None
    if raw_df is None:
        st.sidebar.error("No bundled dataset found. Run `python generate_sample_data.py` "
                          "or place your CSV at data/nepse_manufacturing.csv.")

available_companies = config.COMPANIES
if raw_df is not None:
    available_companies = [c for c in config.COMPANIES if c in raw_df["symbol"].unique()] or \
                           sorted(raw_df["symbol"].unique().tolist())

symbol = st.sidebar.selectbox("Company (symbol)", available_companies,
                               format_func=lambda s: f"{s} — {config.COMPANY_FULL_NAMES.get(s, '')}")

model_choice = st.sidebar.radio("Model", ["LSTM", "GRU", "Compare Both"], index=2)
forecast_days = st.sidebar.slider("Forecast horizon (NEPSE trading days)", 1, 14,
                                   config.FORECAST_HORIZON_DAYS)

st.sidebar.markdown("---")
st.sidebar.caption(
    "NEPSE trades **Sunday–Thursday**; forecasts automatically skip Friday/Saturday."
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 NEPSE Manufacturing Companies — Stock Price Prediction")
st.caption("LSTM vs GRU deep-learning models · Final Year Project Dashboard")

if raw_df is None:
    st.warning("Load a dataset from the sidebar to continue.")
    st.stop()

models_to_show = ["LSTM", "GRU"] if model_choice == "Compare Both" else [model_choice]
models_ready = all(model_available(symbol, m) for m in models_to_show) and scaler_available(symbol)

# ---------------------------------------------------------------------------
# Fallback: train on the fly if models aren't available yet
# ---------------------------------------------------------------------------
if not models_ready:
    st.warning(
        f"No trained model found for **{symbol}** yet. Run `python -m src.train` from the "
        f"project root first (recommended — full 100-epoch training with early stopping), "
        f"or train a quick demo model right here (fewer epochs, faster, lower accuracy)."
    )
    quick_epochs = st.slider("Quick-train epochs (demo only)", 5, 50, 20)
    if st.button(f"🚀 Quick-train LSTM & GRU for {symbol} now"):
        from src import train as train_mod

        with st.spinner(f"Training LSTM & GRU for {symbol} — this can take a few minutes..."):
            ind_df = data_utils.full_preprocess_pipeline(raw_df, symbol)
            ind_df.to_csv(os.path.join(config.DATA_DIR, f"{symbol}_processed.csv"), index=False)
            prepared = data_utils.prepare_company_data(ind_df)

            rows = []
            reports = _load_confusion_reports()
            for mt in ["LSTM", "GRU"]:
                model, history = train_mod.train_one(
                    symbol, mt, prepared, epochs=quick_epochs,
                    batch_size=config.BATCH_SIZE, patience=max(3, quick_epochs // 4),
                    val_split=config.VALIDATION_SPLIT, seed=config.SEED,
                )
                y_true, y_pred, metrics = train_mod.evaluate_one(model, prepared)
                metrics["Symbol"], metrics["Model"] = symbol, mt
                metrics["Epochs_Trained"] = len(history.history["loss"])
                rows.append(metrics)

                labels, cm, report = model_utils.directional_confusion_matrix(y_true, y_pred)
                reports[f"{symbol}_{mt}"] = {"labels": labels, "matrix": cm.tolist(), "report": report}

                model.save(os.path.join(config.MODELS_DIR, f"{symbol}_{mt}.keras"))
                pred_df = pd.DataFrame({"date": prepared["dates_test"], "actual": y_true, "predicted": y_pred})
                pred_df.to_csv(os.path.join(config.OUTPUTS_DIR, f"{symbol}_{mt}_predictions.csv"), index=False)

            with open(os.path.join(config.MODELS_DIR, f"{symbol}_scaler.pkl"), "wb") as f:
                pickle.dump(prepared["scaler"], f)

            existing = _load_metrics_summary()
            new_rows = pd.DataFrame(rows)
            combined = pd.concat([existing, new_rows], ignore_index=True) if existing is not None else new_rows
            combined = combined.drop_duplicates(subset=["Symbol", "Model"], keep="last")
            combined.to_csv(os.path.join(config.METRICS_DIR, "metrics_summary.csv"), index=False)

            with open(os.path.join(config.METRICS_DIR, "confusion_reports.json"), "w") as f:
                json.dump(reports, f, indent=2)

        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Training complete! Refresh below to see results.")
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Load trained artifacts
# ---------------------------------------------------------------------------
scaler = _load_scaler(os.path.join(config.MODELS_DIR, f"{symbol}_scaler.pkl"))
models = {m: _load_keras_model(os.path.join(config.MODELS_DIR, f"{symbol}_{m}.keras")) for m in models_to_show}
metrics_summary = _load_metrics_summary()
confusion_reports = _load_confusion_reports()

tab_overview, tab_perf, tab_confusion, tab_pred, tab_forecast = st.tabs(
    ["🔍 Overview & EDA", "📊 Model Performance", "🧮 Confusion Matrix",
     "📉 Actual vs Predicted", "🔮 1-Week Forecast"]
)

# ---------------------------------------------------------------------------
# Tab 1: Overview & EDA
# ---------------------------------------------------------------------------
with tab_overview:
    sub = raw_df[raw_df["symbol"] == symbol].sort_values("date")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(sub):,}")
    c2.metric("Latest Close (NPR)", f"{sub['close'].iloc[-1]:,.2f}")
    prev_close = sub["close"].iloc[-2] if len(sub) > 1 else sub["close"].iloc[-1]
    delta = sub["close"].iloc[-1] - prev_close
    c3.metric("Last Change", f"{delta:,.2f}", f"{(delta/prev_close*100):.2f}%")
    c4.metric("Date Range", f"{sub['date'].min().date()} → {sub['date'].max().date()}")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                         vertical_spacing=0.05, subplot_titles=(f"{symbol} Close Price", "Volume"))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["close"], name="Close",
                              line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Bar(x=sub["date"], y=sub["volume"], name="Volume",
                          marker_color="#ff7f0e"), row=2, col=1)
    fig.update_layout(height=550, showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Summary statistics"):
        st.dataframe(sub[["open", "high", "low", "close", "volume"]].describe().round(2),
                     use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Model performance
# ---------------------------------------------------------------------------
with tab_perf:
    st.subheader(f"{symbol} — Model Performance")
    if metrics_summary is not None:
        row = metrics_summary[(metrics_summary["Symbol"] == symbol) &
                               (metrics_summary["Model"].isin(models_to_show))]
        if not row.empty:
            st.dataframe(
                row[["Model", "RMSE", "MAE", "MAPE (%)", "Accuracy (%)", "R2", "Directional_Accuracy (%)"]]
                .set_index("Model").round(3),
                use_container_width=True,
            )

            metric_options = ["Accuracy (%)", "RMSE", "MAE", "MAPE (%)", "R2", "Directional_Accuracy (%)"]
            pick = st.selectbox("Compare metric", metric_options, index=0)
            fig = go.Figure(data=[go.Bar(x=row["Model"], y=row[pick],
                                          marker_color=["#1f77b4", "#ff7f0e"][:len(row)])])
            fig.update_layout(title=f"{symbol}: {pick} by Model", height=380)
            st.plotly_chart(fig, use_container_width=True)

            if len(row) == 2:
                best = row.loc[row["Accuracy (%)"].idxmax(), "Model"]
                st.success(f"🏆 Best model for {symbol} by Accuracy: **{best}**")
        else:
            st.info("No stored metrics for this selection yet — retrain to populate this table.")

        st.markdown("---")
        st.subheader("All companies — LSTM vs GRU overview")
        st.dataframe(metrics_summary.round(3), use_container_width=True)
        overall = metrics_summary.groupby("Model")[
            ["RMSE", "MAE", "MAPE (%)", "Accuracy (%)", "R2", "Directional_Accuracy (%)"]
        ].mean().round(3)
        st.markdown("**Overall average performance across all trained companies:**")
        st.dataframe(overall, use_container_width=True)
    else:
        st.info("Run `python -m src.train` (or quick-train from this app) to populate performance metrics.")

# ---------------------------------------------------------------------------
# Tab 3: Confusion matrix
# ---------------------------------------------------------------------------
with tab_confusion:
    st.subheader(f"{symbol} — Directional (Up / Down) Confusion Matrix")
    st.caption(
        "Converts each day's actual and predicted close price into a next-day "
        "direction label, then compares them like a classification problem — "
        "useful for judging whether the model is useful for trading signals, "
        "not just for minimizing price error."
    )
    cols = st.columns(len(models_to_show))
    for col, mt in zip(cols, models_to_show):
        key = f"{symbol}_{mt}"
        with col:
            st.markdown(f"**{mt}**")
            if key in confusion_reports:
                labels = confusion_reports[key]["labels"]
                cm = np.array(confusion_reports[key]["matrix"])
                fig = go.Figure(data=go.Heatmap(
                    z=cm, x=labels, y=labels, colorscale="Blues",
                    text=cm, texttemplate="%{text}", showscale=False))
                fig.update_layout(
                    xaxis_title="Predicted", yaxis_title="Actual",
                    height=380, margin=dict(t=20, b=20),
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)

                report = confusion_reports[key]["report"]
                acc = report.get("accuracy", None)
                if acc is not None:
                    st.metric("Directional classification accuracy", f"{acc*100:.2f}%")
                rep_df = pd.DataFrame(report).T
                rep_df = rep_df[rep_df.index.isin(labels)]
                st.dataframe(rep_df.round(3), use_container_width=True)
            else:
                st.info("No confusion matrix stored yet — retrain to populate.")

# ---------------------------------------------------------------------------
# Tab 4: Actual vs predicted (test set)
# ---------------------------------------------------------------------------
with tab_pred:
    st.subheader(f"{symbol} — Actual vs Predicted Close Price (Held-out Test Set)")
    fig = go.Figure()
    pred_frames = {}
    for mt in models_to_show:
        df_pred = _load_predictions_csv(symbol, mt)
        if df_pred is None:
            continue
        pred_frames[mt] = df_pred
        if "Actual" not in [t.name for t in fig.data]:
            fig.add_trace(go.Scatter(x=df_pred["date"], y=df_pred["actual"],
                                      name="Actual", line=dict(color="black", width=2)))
        fig.add_trace(go.Scatter(x=df_pred["date"], y=df_pred["predicted"],
                                  name=f"{mt} Predicted", opacity=0.85))
    if fig.data:
        fig.update_layout(height=480, yaxis_title="Price (NPR)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Prediction residuals (Actual − Predicted)**")
        resid_fig = go.Figure()
        for mt, df_pred in pred_frames.items():
            resid_fig.add_trace(go.Histogram(x=df_pred["actual"] - df_pred["predicted"],
                                              name=f"{mt} residuals", opacity=0.6, nbinsx=40))
        resid_fig.update_layout(barmode="overlay", height=350, xaxis_title="Residual (NPR)")
        st.plotly_chart(resid_fig, use_container_width=True)
    else:
        st.info("No stored test-set predictions yet — retrain to populate this chart.")

# ---------------------------------------------------------------------------
# Tab 5: 1-week (N-day) forecast on new data
# ---------------------------------------------------------------------------
with tab_forecast:
    st.subheader(f"{symbol} — {forecast_days}-Day Ahead Forecast")
    st.caption(
        "Recursively forecasts future NEPSE trading days (Sun–Thu) using the most "
        "recent available data. Each step recomputes technical indicators from the "
        "growing history and feeds the model's own prediction back in — a standard "
        "approach for multi-step time-series forecasting. The shaded band is an "
        "illustrative uncertainty range based on recent volatility, not a formal "
        "confidence interval."
    )

    processed_path = os.path.join(config.DATA_DIR, f"{symbol}_processed.csv")
    sub_raw = raw_df[raw_df["symbol"] == symbol].sort_values("date")
    history_source = sub_raw[["date", "open", "high", "low", "close", "volume"]]

    if st.button("🔮 Generate forecast"):
        with st.spinner("Forecasting..."):
            forecast_results = {}
            for mt in models_to_show:
                try:
                    forecast_results[mt] = forecast.forecast_next_days(
                        models[mt], scaler, history_source, n_days=forecast_days)
                except ValueError as e:
                    st.error(f"{mt}: {e}")

        if forecast_results:
            sub_recent = sub_raw.tail(60)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sub_recent["date"], y=sub_recent["close"],
                                      name="Recent Actual", line=dict(color="black")))
            colors = {"LSTM": "#1f77b4", "GRU": "#ff7f0e"}
            for mt, fdf in forecast_results.items():
                fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["predicted_close"],
                                          name=f"{mt} Forecast", line=dict(color=colors.get(mt), dash="dash"),
                                          mode="lines+markers"))
                fig.add_trace(go.Scatter(
                    x=list(fdf["date"]) + list(fdf["date"][::-1]),
                    y=list(fdf["upper_band"]) + list(fdf["lower_band"][::-1]),
                    fill="toself", fillcolor=colors.get(mt, "gray"), opacity=0.12,
                    line=dict(width=0), name=f"{mt} uncertainty band", showlegend=True,
                ))
            fig.update_layout(height=500, yaxis_title="Price (NPR)",
                               title=f"{symbol}: {forecast_days}-Day Forecast")
            st.plotly_chart(fig, use_container_width=True)

            for mt, fdf in forecast_results.items():
                st.markdown(f"**{mt} forecast table**")
                disp = fdf.copy()
                disp["date"] = disp["date"].dt.strftime("%Y-%m-%d (%a)")
                disp = disp.rename(columns={
                    "predicted_close": "Predicted Close (NPR)",
                    "lower_band": "Lower band", "upper_band": "Upper band",
                    "day_ahead": "Day #", "date": "Date",
                })
                st.dataframe(disp[["Day #", "Date", "Predicted Close (NPR)", "Lower band", "Upper band"]]
                             .round(2), use_container_width=True, hide_index=True)
                st.download_button(
                    f"Download {mt} forecast CSV",
                    data=fdf.to_csv(index=False).encode("utf-8"),
                    file_name=f"{symbol}_{mt}_forecast.csv", mime="text/csv",
                    key=f"dl_{mt}",
                )

            if len(forecast_results) == 2 and "LSTM" in forecast_results and "GRU" in forecast_results:
                l_last = forecast_results["LSTM"]["predicted_close"].iloc[-1]
                g_last = forecast_results["GRU"]["predicted_close"].iloc[-1]
                last_close = sub_raw["close"].iloc[-1]
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Close", f"{last_close:,.2f}")
                c2.metric(f"LSTM: Day {forecast_days} Forecast", f"{l_last:,.2f}",
                          f"{(l_last-last_close)/last_close*100:.2f}%")
                c3.metric(f"GRU: Day {forecast_days} Forecast", f"{g_last:,.2f}",
                          f"{(g_last-last_close)/last_close*100:.2f}%")

    st.info(
        "💡 Tip: switch **Data source** to *Upload new CSV* in the sidebar to forecast on "
        "brand-new data you've collected since the model was trained, without retraining."
    )

st.markdown("---")
st.caption(
    "Final Year Project — Stock Market Prediction for NEPSE Manufacturing Companies using "
    "LSTM & GRU. Predictions are for academic purposes only and are not financial advice."
)
