# NSE Stock Price Analysis

This repository contains an end-to-end stock price data pipeline for Indian equities, including download, preprocessing, feature engineering, correlation-based stock reduction, window generation for time series modeling, and model training/testing.

## Project Structure

- `stock_price_analysis.ipynb` - Main analysis notebook.
  - Downloads historical NSE stock prices using `yfinance`.
  - Cleans and aligns data with NSE trading days.
  - Computes technical and calendar features.
  - Masks long missing periods and fills short gaps.
  - Performs correlation clustering to reduce similar stocks.
  - Builds rolling windows for multi-step LSTM forecasting.
  - Trains and evaluates a TensorFlow LSTM model.


- `.venv/` - Local Python virtual environment.


## Key Generated Files

These are sample outputs from the notebook pipeline. Your exact file names may vary depending on which cells are run.

- `nse_daily_prices_long_2021_2025.parquet`
- `nse_part2_features_masked.parquet`
- `nse_part2_features_masked_clustered.parquet`
- `part3_chunks/` - Per-symbol window chunk files.
- `nse_lstm_windows.parquet` - Merged LSTM window dataset.
- `lstm_stock_model_dense5.keras` - Trained TensorFlow model (if training is executed).
- `target_scaler.pkl` / `target_scaler_filtered.pkl` - Saved scaler objects.

## Requirements

Install the required packages in the workspace environment.

```bash
pip install pandas numpy yfinance tqdm pandas_market_calendars scipy scikit-learn tensorflow joblib
```

If you use the Jupyter notebook, install any missing notebook packages and run the notebook cells in order.

## Usage

### Notebook

1. Open `stock_price_analysis.ipynb` in Jupyter or VS Code.
2. Run cells sequentially from top to bottom.
3. Adjust filenames and constants if needed, such as `INPUT_FILE`, `OUTPUT_FILE`, and date ranges.

### Script

Run the helper script after `nse_lstm_windows.parquet` has been generated:

```bash
python preprocess_filter_sentinel.py
```

This will create:

- `nse_lstm_windows_filtered.parquet`
- `target_scaler_filtered.pkl`

## Notes

- The notebook is the primary pipeline and contains all major preprocessing, feature engineering, clustering, window generation, and training steps.
- The script is a focused post-processing step for removing sentinel values and scaling target variables.
- The notebook includes sample training/test splits for 2025 H2 evaluation and model forecasting.

## Tips

- If you only want to regenerate input data, start with the download and preprocessing cells that build `nse_part2_features_masked.parquet`.
- If you want to retrain the model, ensure `nse_lstm_windows.parquet` is complete and run the model training section in the notebook.
- Use the `part3_chunks` directory to inspect per-symbol window generation before merging.
