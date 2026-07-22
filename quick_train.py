import json
import os
import sys

NOTEBOOK_PATH = "stock_price_analysis.ipynb"
with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Replace the loss function in the cell source
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "loss=asymmetric_mse" in src:
            src = src.replace("loss=asymmetric_mse", "loss=\"mse\"")
            cell["source"] = [line + "\n" for line in src.split("\n")]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n")

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook patched to use standard MSE.")

venv_site = os.path.abspath(os.path.join('.venv', 'Lib', 'site-packages'))
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)

code_str = "".join(nb["cells"][23]["source"])
globals_dict = {}
print("Retraining model with standard MSE...")
exec(code_str, globals_dict)
print("Training complete!")
