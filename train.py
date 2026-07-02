"""
End-to-end training script.

For every company in config.COMPANIES this script:
  1. Loads & cleans the raw OHLCV data
  2. Builds technical indicators
  3. Prepares windowed train/test sequences (chronological split, MinMax scaled)
  4. Trains an LSTM and a GRU model (with EarlyStopping)
  5. Evaluates on the held-out test set: RMSE, MAE, MAPE, Accuracy, R2,
     Directional Accuracy, and a directional (Up/Down) confusion matrix
  6. Saves: trained models (.keras), the fitted scaler (.pkl), per-company
     metrics (.csv), loss curves, actual-vs-predicted plots, and confusion
     matrix plots into models/ and outputs/

Usage:
    python -m src.train                       # uses data/nepse_manufacturing.csv
    python -m src.train --csv path/to/your.csv --epochs 60
"""
import os
import argparse
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import config, data_utils, model_utils


def set_seeds(seed):
    import tensorflow as tf
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_one(symbol, model_type, prepared, epochs, batch_size, patience, val_split, seed):
    from tensorflow.keras.callbacks import EarlyStopping

    set_seeds(seed)
    input_shape = (prepared["X_train"].shape[1], prepared["X_train"].shape[2])
    model = model_utils.MODEL_BUILDERS[model_type](input_shape)

    early_stop = EarlyStopping(monitor="val_loss", patience=patience,
                                restore_best_weights=True, verbose=0)

    history = model.fit(
        prepared["X_train"], prepared["y_train"],
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
        shuffle=False,  # preserve chronological order
    )
    return model, history


def evaluate_one(model, prepared):
    y_pred_scaled = model.predict(prepared["X_test"], verbose=0).flatten()
    y_pred = data_utils.inverse_transform_target(
        y_pred_scaled, prepared["scaler"], prepared["n_features"], prepared["target_idx"])
    y_true = data_utils.inverse_transform_target(
        prepared["y_test"], prepared["scaler"], prepared["n_features"], prepared["target_idx"])
    metrics = model_utils.compute_metrics(y_true, y_pred)
    return y_true, y_pred, metrics


def save_loss_curve(symbol, model_type, history):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Train Loss")
    ax.plot(history.history["val_loss"], label="Val Loss")
    ax.set_title(f"{symbol} — {model_type} Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (scaled)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f"{symbol}_{model_type}_loss_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_pred_plot(symbol, model_type, dates, y_true, y_pred):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, y_true, label="Actual", color="black", linewidth=1.3)
    ax.plot(dates, y_pred, label=f"{model_type} Predicted", linewidth=1, alpha=0.85)
    ax.set_title(f"{symbol} — {model_type}: Actual vs Predicted Close (Test Set)")
    ax.set_ylabel("Price (NPR)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f"{symbol}_{model_type}_actual_vs_pred.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_confusion_matrix_plot(symbol, model_type, labels, cm):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_xlabel("Predicted Direction")
    ax.set_ylabel("Actual Direction")
    ax.set_title(f"{symbol} — {model_type} Directional Confusion Matrix")
    fig.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f"{symbol}_{model_type}_confusion_matrix.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Train LSTM & GRU models for NEPSE manufacturing companies")
    parser.add_argument("--csv", type=str, default=config.DEFAULT_CSV)
    parser.add_argument("--companies", type=str, nargs="*", default=config.COMPANIES)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--window-size", type=int, default=config.WINDOW_SIZE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    set_seeds(args.seed)

    print(f"Loading raw data from: {args.csv}")
    raw_df = data_utils.load_raw_csv(args.csv)

    all_metrics_rows = []
    all_confusion_reports = {}

    for symbol in args.companies:
        print(f"\n{'='*70}\n{symbol}: preprocessing\n{'='*70}")
        if symbol not in raw_df["symbol"].unique():
            print(f"  WARNING: symbol '{symbol}' not found in CSV — skipping.")
            continue

        ind_df = data_utils.full_preprocess_pipeline(raw_df, symbol)
        if len(ind_df) < args.window_size + 50:
            print(f"  WARNING: only {len(ind_df)} usable rows for {symbol}, need at least "
                  f"~{args.window_size + 50}. Skipping.")
            continue

        prepared = data_utils.prepare_company_data(ind_df, window_size=args.window_size)
        print(f"  X_train: {prepared['X_train'].shape}  X_test: {prepared['X_test'].shape}")

        # Save the fully-cleaned indicator dataframe -- forecast.py uses this
        # as the "recent history" source for live 1-week-ahead predictions.
        ind_df.to_csv(os.path.join(config.DATA_DIR, f"{symbol}_processed.csv"), index=False)

        for model_type in config.MODEL_TYPES:
            print(f"\n--- {symbol} | {model_type} ---")
            model, history = train_one(
                symbol, model_type, prepared,
                epochs=args.epochs, batch_size=args.batch_size,
                patience=args.patience, val_split=config.VALIDATION_SPLIT,
                seed=args.seed,
            )
            n_epochs_ran = len(history.history["loss"])
            print(f"  Stopped at epoch {n_epochs_ran} | "
                  f"train_loss={history.history['loss'][-1]:.6f} | "
                  f"val_loss={history.history['val_loss'][-1]:.6f}")

            y_true, y_pred, metrics = evaluate_one(model, prepared)
            metrics["Symbol"] = symbol
            metrics["Model"] = model_type
            metrics["Epochs_Trained"] = n_epochs_ran
            all_metrics_rows.append(metrics)

            labels, cm, report = model_utils.directional_confusion_matrix(y_true, y_pred)
            all_confusion_reports[f"{symbol}_{model_type}"] = {
                "labels": labels, "matrix": cm.tolist(), "report": report,
            }

            # --- persist everything needed for the streamlit app ---
            model_path = os.path.join(config.MODELS_DIR, f"{symbol}_{model_type}.keras")
            model.save(model_path)

            scaler_path = os.path.join(config.MODELS_DIR, f"{symbol}_scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(prepared["scaler"], f)

            save_loss_curve(symbol, model_type, history)
            save_pred_plot(symbol, model_type, prepared["dates_test"], y_true, y_pred)
            save_confusion_matrix_plot(symbol, model_type, labels, cm)

            # per-symbol/model test predictions, useful for the app & report
            pred_df = pd.DataFrame({
                "date": prepared["dates_test"],
                "actual": y_true,
                "predicted": y_pred,
            })
            pred_df.to_csv(os.path.join(config.OUTPUTS_DIR, f"{symbol}_{model_type}_predictions.csv"),
                            index=False)

            print(f"  Accuracy: {metrics['Accuracy (%)']:.2f}% | RMSE: {metrics['RMSE']:.2f} | "
                  f"Directional Acc: {metrics['Directional_Accuracy (%)']:.2f}%")

    if not all_metrics_rows:
        print("\nNo models were trained (check your --csv / --companies). Exiting.")
        return

    metrics_df = pd.DataFrame(all_metrics_rows)[
        ["Symbol", "Model", "RMSE", "MAE", "MAPE (%)", "Accuracy (%)", "R2",
         "Directional_Accuracy (%)", "Epochs_Trained"]
    ].round(3)
    metrics_path = os.path.join(config.METRICS_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)

    with open(os.path.join(config.METRICS_DIR, "confusion_reports.json"), "w") as f:
        json.dump(all_confusion_reports, f, indent=2)

    print(f"\n{'='*70}\nTraining complete.")
    print(f"Metrics summary saved to: {metrics_path}")
    print(f"Models saved to: {config.MODELS_DIR}")
    print(f"Plots saved to: {config.PLOTS_DIR}")
    print(f"\n{metrics_df.to_string(index=False)}")

    overall = metrics_df.groupby("Model")[
        ["RMSE", "MAE", "MAPE (%)", "Accuracy (%)", "R2", "Directional_Accuracy (%)"]
    ].mean().round(3)
    print(f"\nOverall average performance:\n{overall.to_string()}")


if __name__ == "__main__":
    main()
