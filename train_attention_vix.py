import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import MinMaxScaler
import joblib

INPUT_FILE = "nse_lstm_windows_vix.parquet"
MODEL_FILE = "lstm_stock_model_vix.keras"
SCALER_FILE = "target_scaler_vix.pkl"
LOOKBACK = 30
HORIZON = 5

BASE_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "log_return", "rsi_14", "ema_12", "ema_26", "macd", "macd_signal",
    "sma_20", "sma_50", "dist_sma_20", "dist_sma_50",
    "roc_3", "roc_5", "roc_10", "bb_width", "atr_14", "atr_pct",
    "hist_vol_10", "hist_vol_21", "vol_zscore", "obv", "mfi_14",
    "body_size", "upper_shadow_ratio", "lower_shadow_ratio",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "bench_corr_20", "rel_strength",
    "vix_close", "vix_return", "vix_corr_20"
]

print("Loading dataset...")
df = pd.read_parquet(INPUT_FILE)

df["target_start"] = pd.to_datetime(df["target_start"])
# Split at mid 2025 to leave 6 months for validation
train_mask = df["target_start"] < "2025-06-30"
val_mask = df["target_start"] >= "2025-06-30"

X_cols = [f"{feat}_t{t+1}" for feat in BASE_FEATURES for t in range(LOOKBACK)]
y_cols = [f"y_t{h+1}" for h in range(HORIZON)]

print("Extracting features...")
X_train_raw = df.loc[train_mask, X_cols].values
y_train_raw = df.loc[train_mask, y_cols].values
X_val_raw = df.loc[val_mask, X_cols].values
y_val_raw = df.loc[val_mask, y_cols].values

print("Scaling data...")
feature_scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = feature_scaler.fit_transform(X_train_raw)
X_val_scaled = feature_scaler.transform(X_val_raw)

target_scaler = MinMaxScaler(feature_range=(-1, 1))
y_train = target_scaler.fit_transform(y_train_raw)
y_val = target_scaler.transform(y_val_raw)

joblib.dump(target_scaler, SCALER_FILE)
joblib.dump(feature_scaler, "feature_scaler.pkl")

X_train = X_train_scaled.reshape(-1, LOOKBACK, len(BASE_FEATURES))
X_val = X_val_scaled.reshape(-1, LOOKBACK, len(BASE_FEATURES))

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

print("Building Attention model...")
inputs = layers.Input(shape=(LOOKBACK, len(BASE_FEATURES)))
x = layers.LSTM(64, return_sequences=True,
                kernel_regularizer=regularizers.l2(1e-4),
                recurrent_regularizer=regularizers.l2(1e-4))(inputs)
x = layers.Dropout(0.25)(x)
x = layers.Attention()([x, x])
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(HORIZON, activation="linear", kernel_initializer="he_normal")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(MODEL_FILE, save_best_only=True)
]

print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

print(f"Saved optimized model to {MODEL_FILE}")
print(f"Saved scaler to {SCALER_FILE}")
