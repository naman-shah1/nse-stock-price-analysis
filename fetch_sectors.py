import pandas as pd
import yfinance as yf
import json
import time

CLUSTERED_FILE = "nse_lstm_windows_filtered.parquet"
print("Loading clustered symbols...")
df_clust = pd.read_parquet(CLUSTERED_FILE, columns=["symbol"])
symbols = df_clust["symbol"].dropna().unique().tolist()

from concurrent.futures import ThreadPoolExecutor

def fetch_sector(sym):
    try:
        info = yf.Ticker(f"{sym}.NS").info
        return sym, info.get("sector", "Miscellaneous")
    except:
        return sym, "Miscellaneous"

sectors = {}
print(f"Fetching sectors for {len(symbols)} symbols...")

with ThreadPoolExecutor(max_workers=20) as executor:
    results = executor.map(fetch_sector, symbols)
    for sym, sec in results:
        sectors[sym] = sec

with open("sectors.json", "w") as f:
    json.dump(sectors, f, indent=4)
    
print("Successfully saved sectors.json")
