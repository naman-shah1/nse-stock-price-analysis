import yfinance as yf
import pandas as pd
print("Loading symbols...")
df_clust = pd.read_parquet("nse_lstm_windows_filtered.parquet", columns=["symbol"])
symbols = df_clust["symbol"].dropna().unique().tolist()
tickers = [f"{s}.NS" for s in symbols]

# The simulation period is Jan 1, 2026 to Jun 30, 2026.
# We will download 1h data from Jan 1, 2026 to Jul 5, 2026 to cover the entire period safely.
start_date = "2026-01-01"
end_date = "2026-07-05"

print(f"Downloading 1h data for {len(tickers)} symbols from {start_date} to {end_date}...")

# Download in batches to avoid rate limits
batch_size = 20
all_data = []

for i in range(0, len(tickers), batch_size):
    batch_symbols = tickers[i:i+batch_size]
    print(f"Downloading batch {i//batch_size + 1}...")
    
    # yfinance returns a multi-index dataframe if multiple symbols are passed
    data = yf.download(batch_symbols, start=start_date, end=end_date, interval="1h", group_by="ticker", progress=False)
    
    for sym in batch_symbols:
        if sym in data:
            df = data[sym].copy()
            df = df.dropna()
            if not df.empty:
                df["symbol"] = sym.replace(".NS", "")
                all_data.append(df)

if not all_data:
    print("Failed to download any data.")
else:
    final_df = pd.concat(all_data)
    final_df.to_parquet("nse_hourly_data.parquet")
    print(f"Successfully downloaded and saved {len(final_df)} hourly records for {final_df['symbol'].nunique()} symbols.")
