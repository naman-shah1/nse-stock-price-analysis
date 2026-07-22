import json
import re

with open('stock_price_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # We only want to patch cells that use the scaler or model training / predicting.
        if 'keras' in source.lower() or 'scaler_file' in source.lower() or 'target_scaler' in source.lower():
            # 1. Update INPUT_FILE
            source = source.replace('INPUT_FILE = "nse_lstm_windows.parquet"', 'INPUT_FILE = "nse_lstm_windows_filtered.parquet"')
            
            # 2. Update SCALER_FILE
            source = source.replace('SCALER_FILE = "target_scaler.pkl"', 'SCALER_FILE = "target_scaler_filtered.pkl"')
            source = source.replace('SCALER_FILE = "target_scaler_test.pkl" if TEST_MODE else "target_scaler.pkl"', 'SCALER_FILE = "target_scaler_filtered_test.pkl" if TEST_MODE else "target_scaler_filtered.pkl"')
            
            # 3. Fix the fit_transform blocks
            if 'target_scaler = StandardScaler()' in source:
                # Cell 8
                if 'y_val_scaled = target_scaler.transform' in source:
                    old_block = """target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
joblib.dump(target_scaler, SCALER_FILE)"""
                    new_block = """target_scaler = joblib.load(SCALER_FILE)
y_train_scaled = y_train
y_val_scaled = y_val"""
                    source = source.replace(old_block, new_block)
                
                # Cell 15
                if 'y_test_scaled = target_scaler.transform' in source:
                    old_block = """target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
joblib.dump(target_scaler, SCALER_FILE)"""
                    new_block = """target_scaler = joblib.load(SCALER_FILE)
y_train_scaled = y_train
y_test_scaled = y_test"""
                    source = source.replace(old_block, new_block)
            
            # Update the source array in the cell
            # splitlines(True) keeps the newlines at the end of each line, just like original jupyter cells
            cell['source'] = [line + '\n' for line in source.split('\n')]
            # Fix the last line which shouldn't have a trailing newline if it didn't before
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open('stock_price_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully!")
