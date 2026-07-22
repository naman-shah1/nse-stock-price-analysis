import pandas as pd
import numpy as np
import yfinance as yf

INPUT_FILE = "nse_lstm_windows_filtered.parquet"
OUTPUT_FILE = "nse_lstm_windows_vix.parquet"

print(f"Loading {INPUT_FILE}...")
df = pd.read_parquet(INPUT_FILE)

# Ensure the index is a datetime index if 'date' is a column
if "date" not in df.columns:
    df = df.reset_index()
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
else:
    # If the index was named something else like Date
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

# Find the absolute min and max dates across all stocks
min_date = df["date"].min()
max_date = df["date"].max()
print(f"Dataset date range: {min_date} to {max_date}")

print("Downloading India VIX data...")
vix = yf.download("^INDIAVIX", start=min_date, end=max_date + pd.Timedelta(days=5), auto_adjust=True, progress=False)
vix = vix.reset_index()

date_col = "Date" if "Date" in vix.columns else vix.columns[0]
vix["date"] = pd.to_datetime(vix[date_col]).dt.tz_localize(None)

# Handle multi-level columns if yfinance returns them
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in vix.columns]

close_col = [c for c in vix.columns if "Close" in c][0]

vix = vix[["date", close_col]].rename(columns={close_col: "vix_close"})
vix = vix.set_index("date")

# Forward fill missing days
vix = vix.resample('D').ffill()

vix["vix_return"] = np.log(vix["vix_close"] / vix["vix_close"].shift(1))

print("Merging VIX data into main dataset...")
# Reset index to perform merge
df = df.set_index("date")

# Join VIX data
df = df.join(vix[["vix_close", "vix_return"]], how="left")

# Forward fill any lingering NaNs from the merge, then backward fill
df["vix_close"] = df.groupby("symbol")["vix_close"].ffill().bfill()
df["vix_return"] = df.groupby("symbol")["vix_return"].ffill().bfill()

print("Calculating rolling VIX correlation for each stock...")
df["vix_corr_20"] = df.groupby("symbol").apply(lambda x: x["log_return"].rolling(20).corr(x["vix_return"])).reset_index(level=0, drop=True)

# Fill correlation NaNs
df["vix_corr_20"] = df.groupby("symbol")["vix_corr_20"].ffill().bfill()
df["vix_corr_20"] = df["vix_corr_20"].fillna(0)

# Replace infinities
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=["vix_close", "vix_return", "vix_corr_20"], inplace=True)

df = df.reset_index()

print(f"Saving to {OUTPUT_FILE}...")
df.to_parquet(OUTPUT_FILE)
print("Done!")
