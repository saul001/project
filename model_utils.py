"""
LSTM / GRU model builders, regression metrics, and the directional
(up / down / flat) confusion-matrix helpers used for classification-style
evaluation of what is fundamentally a regression (price) model.
"""
import numpy as np

from src import config


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------
def build_lstm_model(input_shape, lr=None):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    lr = lr or config.LEARNING_RATE
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")
    return model


def build_gru_model(input_shape, lr=None):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    lr = lr or config.LEARNING_RATE
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")
    return model


MODEL_BUILDERS = {"LSTM": build_lstm_model, "GRU": build_gru_model}


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    accuracy = float(100 - mape)
    r2 = float(r2_score(y_true, y_pred))

    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    directional_acc = float(np.mean(true_dir == pred_dir) * 100)

    return {
        "RMSE": rmse, "MAE": mae, "MAPE (%)": mape,
        "Accuracy (%)": accuracy, "R2": r2,
        "Directional_Accuracy (%)": directional_acc,
    }


# ---------------------------------------------------------------------------
# Directional confusion matrix (Up / Down / Flat classification)
# ---------------------------------------------------------------------------
def to_direction_labels(prices, flat_threshold_pct=0.0):
    """Convert a price series into Up/Down/Flat day-over-day direction labels.

    flat_threshold_pct: if the % change magnitude is below this threshold the
    day is labelled 'Flat'. Use 0.0 for a strict binary-style Up/Down split
    (ties go to 'Flat' only on exact equality), or e.g. 0.1 to treat moves
    under 0.1% as noise/flat.
    """
    prices = np.asarray(prices).reshape(-1)
    pct_change = np.diff(prices) / prices[:-1] * 100
    labels = np.where(pct_change > flat_threshold_pct, "Up",
              np.where(pct_change < -flat_threshold_pct, "Down", "Flat"))
    return labels


def directional_confusion_matrix(y_true, y_pred, flat_threshold_pct=0.0):
    """Builds a confusion matrix comparing the ACTUAL next-day direction to
    the PREDICTED next-day direction. Returns (labels, matrix, report_dict).
    """
    from sklearn.metrics import confusion_matrix, classification_report

    true_labels = to_direction_labels(y_true, flat_threshold_pct)
    pred_labels = to_direction_labels(y_pred, flat_threshold_pct)

    present = sorted(set(true_labels) | set(pred_labels),
                      key=lambda x: {"Down": 0, "Flat": 1, "Up": 2}[x])

    cm = confusion_matrix(true_labels, pred_labels, labels=present)
    report = classification_report(true_labels, pred_labels, labels=present,
                                    output_dict=True, zero_division=0)
    return present, cm, report
