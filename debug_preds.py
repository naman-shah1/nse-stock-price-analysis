import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
import joblib

INPUT_FILE = "nse_lstm_windows_vix.parquet"
df = pd.read_parquet(INPUT_FILE)

# grab a sample
sample = df.iloc[-100:]

XGB_MODEL_FILE = "xgb_stock_model.json"
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(XGB_MODEL_FILE)

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

X_cols = [f"{feat}_t30" for feat in BASE_FEATURES]
X_xgb = sample[X_cols].values
xgb_preds = xgb_model.predict(X_xgb)

print("XGBoost Predictions (first 20):")
print(xgb_preds[:20])

MODEL_FILE = "lstm_stock_model_vix.keras"
model = keras.models.load_model(MODEL_FILE)
target_scaler = joblib.load("target_scaler_vix.pkl")

# LSTM features
from sklearn.preprocessing import MinMaxScaler
lstm_preds = []
for i in range(20):
    row = sample.iloc[i]
    hist_raw = []
    for t in range(1, 31):
        t_cols = [f"{feat}_t{t}" for feat in BASE_FEATURES]
        hist_raw.append(row[t_cols].values)
    hist_raw = np.array(hist_raw)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    hist_scaled = scaler.fit_transform(hist_raw)
    
    pred_scaled = model.predict(hist_scaled.reshape(1, 30, 38), verbose=0)
    pred_unscaled = target_scaler.inverse_transform(pred_scaled)
    lstm_preds.append(np.sum(pred_unscaled))

print("\nLSTM Predictions (first 20):")
print(lstm_preds)

print("\nThreshold: -0.04")
