import json
import sys
import os

NOTEBOOK_PATH = "stock_price_analysis.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The old model definition strings to look for in cell 15
old_model_str = """model = keras.Sequential([
    layers.Input(shape=(LOOKBACK, len(base_features))),
    layers.LSTM(64, return_sequences=True,
                kernel_regularizer=regularizers.l2(1e-4),
                recurrent_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.25),
    layers.LSTM(32,
                kernel_regularizer=regularizers.l2(1e-4),
                recurrent_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.25),
    layers.Dense(HORIZON, activation="linear", kernel_initializer="he_normal")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse"
)"""

new_model_str = """import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
from tensorflow.keras import Model

@tf.keras.utils.register_keras_serializable()
def asymmetric_mse(y_true, y_pred):
    sq_err = K.square(y_pred - y_true)
    penalty = K.cast(K.less(y_pred, 0.0) & K.greater(y_true, 0.0), 'float32') * 2.0
    return K.mean(sq_err * (1.0 + penalty))

inputs = Input(shape=(LOOKBACK, len(base_features)))
x = LSTM(64, return_sequences=True,
         kernel_regularizer=regularizers.l2(1e-4),
         recurrent_regularizer=regularizers.l2(1e-4))(inputs)
x = Dropout(0.25)(x)
x = Attention()([x, x])
x = GlobalAveragePooling1D()(x)
outputs = Dense(HORIZON, activation="linear", kernel_initializer="he_normal")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=asymmetric_mse
)"""

patched_cells = 0
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "model = keras.Sequential([" in source:
            # We found the block
            # Since the block might have slight formatting differences between Cell 8 and 15 (e.g. Masking layer),
            # we'll use string slice patching based on bounds.
            start_idx = source.find("model = keras.Sequential([")
            end_idx = source.find("loss=\"mse\"\n)")
            
            if end_idx != -1:
                source = source[:start_idx] + new_model_str + source[end_idx + 12:]
                cell["source"] = [line + "\n" for line in source.split("\n")]
                if cell["source"]:
                    cell["source"][-1] = cell["source"][-1].rstrip("\n")
                patched_cells += 1

print(f"Patched {patched_cells} cells.")

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook saved.")

# Now execute ONLY Cell 15 to train the full model on all data and save it.
globals_dict = {}
venv_site = os.path.abspath(os.path.join('.venv', 'Lib', 'site-packages'))
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)

# Cell 15 is at index 23
code_str = "".join(nb["cells"][23]["source"])
print("Training new model (Cell 15)...")
exec(code_str, globals_dict)
print("Model training complete and saved.")
