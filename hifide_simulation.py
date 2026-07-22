import os
import time
import joblib
import warnings
import json
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import pandas_market_calendars as mcal
import xgboost as xgb

warnings.filterwarnings("ignore")

# --- Configuration ---
PURSE_START = 500000.0  # 5 Lakhs
MAX_ALLOC_PER_SECTOR = 0.30  # Increased to 30% to accommodate larger high-conviction sizes
TAKE_PROFIT = 0.10  # +10%

START_DATE = "2026-05-05" # Strictly within 60 days of July 1
END_DATE = "2026-07-02"
LOOKBACK = 30
HORIZON = 5

# We sweep across two thresholds (0.005 and 0.01) and two TP modes (Infinite vs 10%)
STRATEGIES = {
    "T_0.005_Trailing_7%_NoTP": {"threshold": 0.005, "trailing_sl": 0.07, "tp": float('inf')},
    "T_0.005_Trailing_7%_10%TP": {"threshold": 0.005, "trailing_sl": 0.07, "tp": 0.10},
    "T_0.010_Trailing_7%_NoTP": {"threshold": 0.010, "trailing_sl": 0.07, "tp": float('inf')},
    "T_0.010_Trailing_7%_10%TP": {"threshold": 0.010, "trailing_sl": 0.07, "tp": 0.10},
}

MODEL_FILE = "lstm_stock_model_vix.keras"
XGB_MODEL_FILE = "xgb_stock_model.json"
SCALER_FILE = "target_scaler_vix.pkl"
FEATURE_SCALER_FILE = "feature_scaler.pkl"
CLUSTERED_FILE = "nse_lstm_windows_filtered.parquet"

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

print("Loading clustered symbols and sectors...")
df_clust = pd.read_parquet(CLUSTERED_FILE, columns=["symbol"])
symbols = df_clust["symbol"].dropna().unique().tolist()
tickers = [f"{s}.NS" for s in symbols]

with open("sectors.json", "r") as f:
    sectors_map = json.load(f)

print("Initializing empty intraday cache for lazy-loading 1m data...")
intraday_cache = {}

print(f"Downloading daily data for {len(symbols)} symbols...")
# Need at least 200 calendar days to guarantee 100 trading days for the 50-day SMA computation
dl_start = (pd.to_datetime(START_DATE) - pd.Timedelta(days=200)).strftime('%Y-%m-%d')
raw_data = yf.download(tickers, start=dl_start, end="2026-07-05", group_by="ticker", auto_adjust=True, progress=False)
bench = yf.download("^NSEI", start=dl_start, end="2026-07-05", auto_adjust=True, progress=False)
vix = yf.download("^INDIAVIX", start=dl_start, end="2026-07-05", auto_adjust=True, progress=False)

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
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df["date"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        df["symbol"] = sym
        all_stock_dfs[sym] = df.set_index("date")

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
for sym, df in all_stock_dfs.items():
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

print("Loading model and scaler...")
model = keras.models.load_model(MODEL_FILE)
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(XGB_MODEL_FILE)
target_scaler = joblib.load(SCALER_FILE)

nse = mcal.get_calendar("NSE")
schedule = nse.schedule(start_date=START_DATE, end_date=END_DATE)
trading_days = pd.DatetimeIndex(schedule.index).tz_localize(None)

portfolios = {k: {"purse": PURSE_START, "holdings": {}, "trades": []} for k in STRATEGIES.keys()}

print("Starting Walk-Forward Diversified Risk-Managed Simulation...")
for i, current_day in enumerate(tqdm(trading_days)):
    if i == len(trading_days) - 1:
        break 
        
    next_day = trading_days[i+1]
        
    day_predictions = []
    
    for sym, df in all_stock_dfs.items():
        hist = df[df.index < current_day].copy()
        if len(hist) < LOOKBACK:
            continue
            
        try:
            curr_open = df.loc[current_day, "Open"]
            curr_high = df.loc[current_day, "High"]
            curr_low = df.loc[current_day, "Low"]
            curr_close = df.loc[current_day, "Close"]
        except KeyError:
            continue
            
        # Re-introduce per-stock on-the-fly scaling (original robust design)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        hist_scaled = scaler.fit_transform(hist[BASE_FEATURES])
        features_scaled = hist_scaled[-LOOKBACK:]
        
        # XGBoost expects unscaled latest features
        latest_unscaled = hist[BASE_FEATURES].values[-1]
        
        day_predictions.append({
            "symbol": sym,
            "features_scaled": features_scaled,
            "latest_unscaled": latest_unscaled,
            "curr_open": curr_open,
            "curr_high": curr_high,
            "curr_low": curr_low,
            "curr_close": curr_close,
            "sector": sectors_map.get(sym, "Miscellaneous")
        })
        
    if not day_predictions:
        continue
        
    X_batch_scaled = np.array([item["features_scaled"] for item in day_predictions])
    preds_scaled = model.predict(X_batch_scaled, verbose=0)
    preds = target_scaler.inverse_transform(preds_scaled)
    
    # XGBoost Prediction
    X_latest_unscaled = np.array([item["latest_unscaled"] for item in day_predictions])
    xgb_preds = xgb_model.predict(X_latest_unscaled)
    
    for j, p in enumerate(day_predictions):
        p["lstm_pred"] = np.sum(preds[j])
        p["lstm_pred_seq"] = preds[j]
        p["xgb_pred"] = xgb_preds[j]
        
    for strat_name, strat_config in STRATEGIES.items():
        state = portfolios[strat_name]
        thresh = strat_config["threshold"]
        trailing_sl_pct = strat_config["trailing_sl"]
        tp_pct = strat_config["tp"]
        
        total_portfolio_value = state["purse"]
        for holding in state["holdings"].values():
            total_portfolio_value += holding["shares"] * holding["last_close"]
            
        # -- SELL LOGIC (RISK MANAGEMENT & TRAILING STOP) --
        symbols_to_remove = []
        for sym, holding in state["holdings"].items():
            pred_obj = next((p for p in day_predictions if p["symbol"] == sym), None)
            if not pred_obj:
                continue
                
            holding["days_held"] = holding.get("days_held", 0) + 1
            # Update the highest price seen since buying
            holding["highest_price"] = max(holding["highest_price"], pred_obj["curr_high"])
                
            buy_price = holding["buy_price"]
            tp_price = buy_price * (1 + tp_pct)
            sl_price = holding["highest_price"] * (1 - trailing_sl_pct) # Initial TRAILING STOP for the day
            
            sell_price = None
            reason = None
            sell_time = None
            
            # Lazy load 2m data if not present
            if sym not in intraday_cache:
                print(f"Lazy-loading 2m data for {sym}...")
                dl = yf.download(f"{sym}.NS", start=START_DATE, end=END_DATE, interval="2m", progress=False)
                if not dl.empty:
                    dl = dl.reset_index()
                    date_col = "Datetime" if "Datetime" in dl.columns else dl.columns[0]
                    dl["date"] = pd.to_datetime(dl[date_col]).dt.tz_localize(None)
                    dl = dl.set_index("date")
                    intraday_cache[sym] = dl
                else:
                    intraday_cache[sym] = pd.DataFrame()
            
            symbol_1m = intraday_cache[sym]
            
            # Filter 2m data for this exact day
            day_str = current_day.strftime('%Y-%m-%d')
            if symbol_1m.empty:
                symbol_today = pd.DataFrame()
            else:
                mask = (symbol_1m.index.strftime('%Y-%m-%d') == day_str)
                symbol_today = symbol_1m[mask].sort_index()
            
            if symbol_today.empty:
                # Fallback to daily check if 1m data is missing for some reason
                if pred_obj["curr_open"] >= tp_price:
                    sell_price = pred_obj["curr_open"]
                    reason = "TP Hit at Open"
                elif pred_obj["curr_open"] <= sl_price:
                    sell_price = pred_obj["curr_open"]
                    reason = "Trailing SL Hit at Open"
                elif pred_obj["curr_high"] >= tp_price:
                    sell_price = tp_price
                    reason = "TP Hit Intraday (Fallback)"
                elif pred_obj["curr_low"] <= sl_price:
                    sell_price = sl_price
                    reason = "Trailing SL Hit Intraday (Fallback)"
            else:
                for time_idx, row in symbol_today.iterrows():
                    r_high = float(row["High"].iloc[0]) if isinstance(row["High"], pd.Series) else float(row["High"])
                    r_low = float(row["Low"].iloc[0]) if isinstance(row["Low"], pd.Series) else float(row["Low"])
                    
                    # Update trailing stop dynamically minute-by-minute
                    holding["highest_price"] = max(holding["highest_price"], r_high)
                    dynamic_sl = holding["highest_price"] * (1 - trailing_sl_pct)
                    
                    if r_high >= tp_price:
                        sell_price = tp_price
                        reason = f"TP Hit"
                        sell_time = time_idx.strftime('%Y-%m-%d %H:%M:%S')
                        break
                    elif r_low <= dynamic_sl:
                        sell_price = dynamic_sl
                        reason = f"Trailing SL Hit"
                        sell_time = time_idx.strftime('%Y-%m-%d %H:%M:%S')
                        break
                
            if sell_price is not None:
                profit = (sell_price - buy_price) * holding["shares"]
                state["purse"] += (sell_price * holding["shares"])
                
                state["trades"].append({
                    "strategy": strat_name,
                    "symbol": sym,
                    "sector": holding["sector"],
                    "buy_date": holding["buy_date"],
                    "buy_time": holding.get("buy_time", "N/A"),
                    "buy_price": holding["buy_price"],
                    "sell_date": current_day.strftime('%Y-%m-%d'),
                    "sell_time": sell_time if sell_time else "N/A",
                    "sell_price": sell_price,
                    "shares": holding["shares"],
                    "profit": profit,
                    "roi_pct": (sell_price - buy_price) / buy_price * 100,
                    "reason": reason
                })
                symbols_to_remove.append(sym)
            else:
                holding["last_close"] = pred_obj["curr_close"]
                
        for sym in symbols_to_remove:
            del state["holdings"][sym]
            
        # -- BUY LOGIC (DIVERSIFICATION) --
        sector_allocations = {}
        for h in state["holdings"].values():
            sec = h["sector"]
            sector_allocations[sec] = sector_allocations.get(sec, 0) + (h["shares"] * h["last_close"])
            
        max_alloc_sector = total_portfolio_value * MAX_ALLOC_PER_SECTOR
        
        buy_candidates = []
        for pred_obj in day_predictions:
            if pred_obj["symbol"] not in state["holdings"]:
                # ENSEMBLE LOGIC: Both LSTM and XGBoost must agree it crosses the threshold
                if pred_obj["lstm_pred"] > thresh and pred_obj["xgb_pred"] > thresh:
                    buy_candidates.append(pred_obj)
        
        # Sort candidates by the average of their predictions
        buy_candidates.sort(key=lambda x: (x["lstm_pred"] + x["xgb_pred"]) / 2.0, reverse=True)
        
        for p in buy_candidates:
            sec = p["sector"]
            curr_sec_alloc = sector_allocations.get(sec, 0)
            
            if curr_sec_alloc >= max_alloc_sector:
                continue
                
            # Dynamic Confidence-Based Allocation
            # Base 10%, max 25%. Scales with (prediction - threshold) * 10
            dynamic_pct = min(0.25, max(0.05, 0.10 + (p["lstm_pred"] - thresh) * 10))
            
            allocation = state["purse"] * dynamic_pct
            
            # Cap allocation so it doesn't violate sector limits or stock limits
            dynamic_max_stock = total_portfolio_value * dynamic_pct
            remaining_sec_allowance = max_alloc_sector - curr_sec_alloc
            allocation = min(allocation, remaining_sec_allowance, dynamic_max_stock)
            
            if allocation < 1000:
                continue # Not enough to buy 1 share
                
            # Lazy load 2m data for the buy execution
            sym = p["symbol"]
            if sym not in intraday_cache:
                print(f"Lazy-loading 2m data for {sym}...")
                dl = yf.download(f"{sym}.NS", start=START_DATE, end=END_DATE, interval="2m", progress=False)
                if not dl.empty:
                    dl = dl.reset_index()
                    date_col = "Datetime" if "Datetime" in dl.columns else dl.columns[0]
                    dl["date"] = pd.to_datetime(dl[date_col]).dt.tz_localize(None)
                    dl = dl.set_index("date")
                    intraday_cache[sym] = dl
                else:
                    intraday_cache[sym] = pd.DataFrame()
            
            # Execute buy at exactly 09:16 AM market price
            day_str = current_day.strftime('%Y-%m-%d')
            if intraday_cache[sym].empty:
                symbol_today = pd.DataFrame()
            else:
                symbol_today = intraday_cache[sym][intraday_cache[sym].index.strftime('%Y-%m-%d') == day_str].sort_index()
            
            buy_price = p["curr_open"]
            if isinstance(buy_price, pd.Series):
                buy_price = buy_price.iloc[0]
            buy_price = float(buy_price)
            
            buy_time = None
            if not symbol_today.empty:
                # Get the first candle on or after 09:16
                time_mask = symbol_today.index.strftime('%H:%M') >= "09:16"
                if time_mask.any():
                    first_candle = symbol_today[time_mask].iloc[0]
                    bp = first_candle["Close"]
                    if isinstance(bp, pd.Series):
                        bp = bp.iloc[0]
                    buy_price = float(bp)
                    buy_time = symbol_today[time_mask].index[0].strftime('%Y-%m-%d %H:%M:%S')
                
            shares_to_buy = int(allocation // buy_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * buy_price
                state["purse"] -= cost
                sector_allocations[sec] = curr_sec_alloc + cost
                
                state["holdings"][sym] = {
                    "buy_date": current_day.strftime('%Y-%m-%d'),
                    "buy_time": buy_time if buy_time else "N/A",
                    "buy_price": buy_price,
                    "highest_price": buy_price, # Initialize trailing peak
                    "last_close": p["curr_close"],
                    "shares": shares_to_buy,
                    "sector": sec
                }
                    
            if state["purse"] < 1000:
                break

# Close out all remaining positions
for strat_name, state in portfolios.items():
    for sym, holding in list(state["holdings"].items()):
        profit = (holding["last_close"] - holding["buy_price"]) * holding["shares"]
        state["purse"] += (holding["last_close"] * holding["shares"])
        state["trades"].append({
            "strategy": strat_name,
            "symbol": sym,
            "sector": holding["sector"],
            "buy_date": holding["buy_date"],
            "buy_time": holding.get("buy_time", "N/A"),
            "buy_price": holding["buy_price"],
            "sell_date": END_DATE,
            "sell_time": "N/A",
            "sell_price": holding["last_close"],
            "shares": holding["shares"],
            "profit": profit,
            "roi_pct": (holding["last_close"] - holding["buy_price"]) / holding["buy_price"] * 100,
            "reason": "End of Simulation"
        })
    state["holdings"].clear()

print("\n--- DIVERSIFIED SIMULATION RESULTS (Jan 1 2026 - Jun 30 2026) ---")
all_trades = []
for strat_name, state in portfolios.items():
    final_value = state["purse"]
    roi = ((final_value - PURSE_START) / PURSE_START) * 100
    trades = pd.DataFrame(state["trades"])
    
    win_rate = 0
    total_trades = len(trades)
    if total_trades > 0:
        win_rate = (len(trades[trades["profit"] > 0]) / total_trades) * 100
        
    print(f"\n[{strat_name} Strategy]")
    print(f"Final Value: {final_value:,.2f} INR")
    print(f"ROI:         {roi:,.2f}%")
    print(f"Total Trades:{total_trades}")
    print(f"Win Rate:    {win_rate:,.2f}%")
    
    if total_trades > 0:
        print(f"Avg Profit per Winning Trade: {trades[trades['profit'] > 0]['profit'].mean():,.2f}")
        print(f"Avg Loss per Losing Trade:    {trades[trades['profit'] < 0]['profit'].mean():,.2f}")
        
    all_trades.extend(state["trades"])
    
trades_df = pd.DataFrame(all_trades)
trades_df.to_csv("hifide_trades.csv", index=False)
print("\nDetailed trades saved to hifide_trades.csv")
