import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Paths
INPUT_FILE = 'nse_lstm_windows.parquet'
OUTPUT_FILE = 'nse_lstm_windows_filtered.parquet'
SCALER_FILE = 'target_scaler_filtered.pkl'

# Sentinel value used for masking missing targets
SENTINEL = -99.0

# Load data
df = pd.read_parquet(INPUT_FILE)
# Ensure columns aren't duplicated
df = df.loc[:, ~df.columns.duplicated()].copy()

# Identify target columns (y_t1...y_t5)
HORIZON = 5
target_cols = [f'y_t{i}' for i in range(1, HORIZON + 1)]

# Convert target columns to numeric (in case of stray strings)
for c in target_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Drop rows where any target equals the sentinel (masked) value
mask = df[target_cols].apply(lambda col: np.isclose(col, SENTINEL)).any(axis=1)
filtered_df = df[~mask].reset_index(drop=True)
print(f"Original rows: {len(df)}, after sentinel filter: {len(filtered_df)}")

# Fit scaler on filtered training targets only (train split defined by window_end <= TRAIN_END)
TRAIN_END = pd.Timestamp('2025-06-30')
train_df = filtered_df[filtered_df['window_end'] <= TRAIN_END]
# Stack all target values for scaler fitting
train_targets = train_df[target_cols].values.reshape(-1, 1)
scaler = StandardScaler()
scaler.fit(train_targets)
# Save scaler
joblib.dump(scaler, SCALER_FILE)
print(f"Saved scaler to {SCALER_FILE}")

# Apply scaler to all target columns in the filtered dataframe
scaled_targets = scaler.transform(filtered_df[target_cols].values.reshape(-1, 1)).reshape(-1, len(target_cols))
for i, c in enumerate(target_cols):
    filtered_df[c] = scaled_targets[:, i]

# Save the filtered and scaled windows for downstream training
filtered_df.to_parquet(OUTPUT_FILE, index=False)
print(f"Saved filtered windows to {OUTPUT_FILE}")
