import json
import sys
import io
import traceback
import time
import os

# Reconfigure encoding to avoid UnicodeEncodeError on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

NOTEBOOK_PATH = 'stock_price_analysis.ipynb'

print("Loading notebook...")
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Patching the cells in memory first
cells_to_patch = {
    12: "Cell 8",
    14: "Cell 9",
    16: "Cell 10",
    23: "Cell 15"
}

print("Patching notebook cells...")
for idx, label in cells_to_patch.items():
    cell = nb['cells'][idx]
    source = "".join(cell['source'])
    
    # 1. Update input file and scaler file paths
    source = source.replace('INPUT_FILE = "nse_lstm_windows.parquet"', 'INPUT_FILE = "nse_lstm_windows_filtered.parquet"')
    source = source.replace('df = pd.read_parquet(INPUT_FILE)', 'df = pd.read_parquet(INPUT_FILE)\nif "next_date" not in df.columns: df["next_date"] = pd.NaT')
    source = source.replace('SCALER_FILE = "target_scaler.pkl"', 'SCALER_FILE = "target_scaler_filtered.pkl"')
    source = source.replace(
        'SCALER_FILE = "target_scaler_test.pkl" if TEST_MODE else "target_scaler.pkl"',
        'SCALER_FILE = "target_scaler_filtered_test.pkl" if TEST_MODE else "target_scaler_filtered.pkl"'
    )
    
    # 2. Replaces standard scaler fit_transform with loading the pre-fitted scaler
    old_fit_block_8 = """target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
joblib.dump(target_scaler, SCALER_FILE)"""
    new_fit_block_8 = """target_scaler = joblib.load(SCALER_FILE)
y_train_scaled = y_train
y_val_scaled = y_val"""
    source = source.replace(old_fit_block_8, new_fit_block_8)
    
    old_fit_block_15 = """target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
joblib.dump(target_scaler, SCALER_FILE)"""
    new_fit_block_15 = """target_scaler = joblib.load(SCALER_FILE)
y_train_scaled = y_train
y_test_scaled = y_test"""
    source = source.replace(old_fit_block_15, new_fit_block_15)
    
    # 3. Bugfix: Inverse transform the scaled 'true' values to compute metrics in the raw/unscaled domain
    # In Cell 8 (Index 12)
    source = source.replace(
        "true = y_val\n",
        "true = target_scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(y_val.shape)\n"
    )
    # In Cell 9 (Index 14)
    source = source.replace(
        "true = y_test\n",
        "true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)\n"
    )
    # In Cell 15 (Index 23)
    source = source.replace(
        "true = y_test\n",
        "true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)\n"
    )
    
    # Update the cell source
    cell['source'] = [line + '\n' for line in source.split('\n')]
    if cell['source']:
        cell['source'][-1] = cell['source'][-1].rstrip('\n')

# Save the patched notebook before running
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook patched and saved successfully.")

# Setup execution environment
globals_dict = {}
# Add python path of .venv/Lib/site-packages to sys.path just to be safe
venv_site = os.path.abspath(os.path.join('.venv', 'Lib', 'site-packages'))
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)

execution_indices = [12, 14, 16, 23]
execution_count = 1

for idx in execution_indices:
    cell = nb['cells'][idx]
    code_str = "".join(cell['source'])
    label = cells_to_patch[idx]
    
    print(f"\n--- Executing {label} (Notebook Index {idx}) ---")
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    redirected_error = io.StringIO()
    sys.stdout = redirected_output
    sys.stderr = redirected_error
    
    start_time = time.time()
    exception = None
    try:
        exec(code_str, globals_dict)
    except Exception as e:
        exception = e
        traceback.print_exc(file=redirected_error)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
    duration = time.time() - start_time
    stdout_val = redirected_output.getvalue()
    stderr_val = redirected_error.getvalue()
    
    print(f"Completed in {duration:.2f} seconds.")
    if stdout_val:
        print("Stdout:\n" + stdout_val.strip())
    if stderr_val:
        print("Stderr:\n" + stderr_val.strip())
        
    if exception:
        print(f"ERROR: Execution of {label} failed!")
        # Save notebook with failure info
        cell['outputs'] = [{
            "output_type": "error",
            "ename": type(exception).__name__,
            "evalue": str(exception),
            "traceback": stderr_val.split('\n')
        }]
        cell['execution_count'] = execution_count
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        sys.exit(1)
        
    # Format and store outputs in notebook cell
    outputs = []
    if stdout_val:
        outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": [line + '\n' for line in stdout_val.split('\n')]
        })
    if stderr_val:
        outputs.append({
            "output_type": "stream",
            "name": "stderr",
            "text": [line + '\n' for line in stderr_val.split('\n')]
        })
        
    cell['outputs'] = outputs
    cell['execution_count'] = execution_count
    execution_count += 1

print("\nAll cells executed successfully!")
print("Saving final executed notebook...")
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Finished!")
