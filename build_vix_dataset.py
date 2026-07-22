import os
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

CLUSTERED_FILE = "nse_lstm_windows_filtered.parquet"
OUTPUT_FILE = "nse_lstm_windows_vix.parquet"
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

print("Loading clustered symbols...")
df_clust = pd.read_parquet(CLUSTERED_FILE, columns=["symbol"])
symbols = df_clust["symbol"].dropna().unique().tolist()
tickers = [f"{s}.NS" for s in symbols]

print(f"Downloading data for {len(symbols)} symbols from 2021-01-01 to 2025-12-31...")
raw_data = yf.download(tickers, start="2021-01-01", end="2025-12-31", group_by="ticker", auto_adjust=True, progress=True, threads=True)
bench = yf.download("^NSEI", start="2021-01-01", end="2025-12-31", auto_adjust=True, progress=False)
vix = yf.download("^INDIAVIX", start="2021-01-01", end="2025-12-31", auto_adjust=True, progress=False)

def prep_index(df_):
    df_ = df_.reset_index()
    date_col = "Date" if "Date" in df_.columns else df_.columns[0]
    df_["date"] = pd.to_datetime(df_[date_col]).dt.tz_localize(None)
    df_ = df_.set_index("date")
    return df_

bench = prep_index(bench)
bench = bench[["Close"]].rename(columns={"Close": "nifty_close"})
bench["nifty_return"] = np.log(bench["nifty_close"] / bench["nifty_close"].shift(1))

vix = prep_index(vix)
# Handle multi-level columns if yfinance returns them
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in vix.columns]
close_col = [c for c in vix.columns if "Close" in c][0]
vix = vix[[close_col]].rename(columns={close_col: "vix_close"})
vix = vix.resample('D').ffill()
vix["vix_return"] = np.log(vix["vix_close"] / vix["vix_close"].shift(1))

all_stock_dfs = {}
for sym in symbols:
    ticker = f"{sym}.NS"
    if ticker in raw_data.columns.levels[0]:
        df = raw_data[ticker].copy()
        df = df.dropna(subset=["Close"])
        if len(df) < 100:
            continue
        df = prep_index(df)
        df["symbol"] = sym
        all_stock_dfs[sym] = df

print(f"Valid stocks downloaded: {len(all_stock_dfs)}")

def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def atr(df_):
    tr = pd.concat([df_["High"] - df_["Low"], (df_["High"] - df_["Close"].shift()).abs(), (df_["Low"] - df_["Close"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()
def mfi(df_, window=14):
    tp = (df_["High"] + df_["Low"] + df_["Close"]) / 3
    mf = tp * df_["Volume"]
    delta_tp = tp.diff()
    mfr = mf.where(delta_tp > 0, 0.0).rolling(window).sum() / mf.where(delta_tp < 0, 0.0).abs().rolling(window).sum()
    return 100 - (100 / (1 + mfr))

print("Computing features...")
for sym, df in tqdm(all_stock_dfs.items()):
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    
    df["rsi_14"] = rsi(df["Close"], 14)
    df["ema_12"] = ema(df["Close"], 12)
    df["ema_26"] = ema(df["Close"], 26)
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = ema(df["macd"], 9)
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["dist_sma_20"] = (df["Close"] / df["sma_20"]) - 1
    df["dist_sma_50"] = (df["Close"] / df["sma_50"]) - 1
    for n in [3, 5, 10]: df[f"roc_{n}"] = df["Close"].pct_change(n)
    
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_width"] = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / bb_mid
    df["atr_14"] = atr(df)
    df["atr_pct"] = df["atr_14"] / df["Close"]
    df["hist_vol_10"] = df["log_return"].rolling(10).std()
    df["hist_vol_21"] = df["log_return"].rolling(21).std()
    
    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    df["vol_zscore"] = (df["Volume"] - vol_mean) / vol_std
    
    obv = [0.0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]: obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]: obv.append(obv[-1] - df["Volume"].iloc[i])
        else: obv.append(obv[-1])
    df["obv"] = obv
    df["mfi_14"] = mfi(df, 14)
    
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    df["body_size"] = (df["Close"] - df["Open"]) / rng
    df["upper_shadow_ratio"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / rng
    df["lower_shadow_ratio"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / rng
    
    bench_aligned = bench.reindex(df.index).ffill().bfill()
    df["nifty_close"] = bench_aligned["nifty_close"]
    df["nifty_return"] = bench_aligned["nifty_return"]
    df["rel_strength"] = df["Close"] / df["nifty_close"]
    df["bench_corr_20"] = df["log_return"].rolling(20).corr(df["nifty_return"])
    
    vix_aligned = vix.reindex(df.index).ffill().bfill()
    df["vix_close"] = vix_aligned["vix_close"]
    df["vix_return"] = vix_aligned["vix_return"]
    df["vix_corr_20"] = df["log_return"].rolling(20).corr(df["vix_return"])
    
    for c in BASE_FEATURES:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
            
    df = df.dropna(subset=BASE_FEATURES)
    all_stock_dfs[sym] = df

print("Creating windows...")
windowed_records = []
for sym, df in tqdm(all_stock_dfs.items()):
    dates = df.index
    values = df[BASE_FEATURES].values
    log_returns = df["log_return"].values
    
    for i in range(LOOKBACK, len(df) - HORIZON + 1):
        window_dates = dates[i-LOOKBACK:i]
        window_vals = values[i-LOOKBACK:i]
        
        target_dates = dates[i:i+HORIZON]
        target_returns = log_returns[i:i+HORIZON]
        
        record = {
            "symbol": sym,
            "start_date": dates[0],
            "window_start": window_dates[0],
            "window_end": window_dates[-1],
            "target_start": target_dates[0],
            "target_end": target_dates[-1]
        }
        
        for feat_idx, feat_name in enumerate(BASE_FEATURES):
            for t in range(LOOKBACK):
                record[f"{feat_name}_t{t+1}"] = window_vals[t, feat_idx]
                
        for h in range(HORIZON):
            record[f"y_t{h+1}"] = target_returns[h]
            
        windowed_records.append(record)

print(f"Generated {len(windowed_records)} windows.")
df_windows = pd.DataFrame(windowed_records)
print(f"Saving to {OUTPUT_FILE}...")
df_windows.to_parquet(OUTPUT_FILE)
print("Done!")
