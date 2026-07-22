import json

with open('stock_price_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = []
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        first_line = cell['source'][0].strip() if cell['source'] else 'EMPTY'
        code_cells.append((idx, first_line, source[:100]))

print(f"Total cells: {len(nb['cells'])}")
print(f"Total code cells: {len(code_cells)}")
for i, (idx, first_line, snippet) in enumerate(code_cells):
    print(f"Code Cell {i} (Notebook Index {idx}): {first_line} | Snippet: {repr(snippet)}")
