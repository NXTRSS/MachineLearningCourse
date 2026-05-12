#!/usr/bin/env python3
"""
Execute Function_Calling_TEST.ipynb with allow_errors=True.
Saves executed notebook and reports cell-by-cell results.
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NB_PATH = "Function_Calling_TEST.ipynb"
OUTPUT_PATH = "Function_Calling_TEST_executed.ipynb"

print(f"Loading {NB_PATH}...")
nb = nbformat.read(NB_PATH, as_version=4)

# Mark Gradio cell to skip (would block forever)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        src = cell.source.strip()
        if "launch_chat" in src or "demo.launch" in src:
            print(f"  Skipping Cell {i} (Gradio launch)")
            cell.source = f"# SKIPPED: {src.split(chr(10))[0]}\nprint('Gradio launch skipped')"

print("Executing notebook (allow_errors=True, timeout=300s per cell)...")
ep = ExecutePreprocessor(
    timeout=300,         # 5 min per cell (agent loops need it)
    kernel_name="ml",    # match notebook's kernel
    allow_errors=True,   # continue past errors!
)

try:
    ep.preprocess(nb, {"metadata": {"path": "."}})
    print("Notebook execution complete!")
except Exception as e:
    print(f"Execution stopped: {type(e).__name__}: {str(e)[:300]}")

# Save executed notebook
nbformat.write(nb, OUTPUT_PATH)
print(f"\nSaved: {OUTPUT_PATH}")

# Analyze cell-by-cell
print(f"\n{'='*60}")
print("CELL-BY-CELL RESULTS:")
errors = []
ok_count = 0
skip_count = 0

for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = cell.source.strip()
    if not src:
        continue

    first_line = src.split("\n")[0][:80]

    has_error = False
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            has_error = True
            ename = output.get("ename", "?")
            evalue = output.get("evalue", "?")[:200]
            errors.append((i, first_line, f"{ename}: {evalue}"))
            print(f"  Cell {i:3d}: ERROR  {ename}: {evalue[:100]}")
            break

    if not has_error:
        exec_count = cell.get("execution_count")
        if exec_count is not None:
            ok_count += 1
            print(f"  Cell {i:3d}: OK [{exec_count:2d}] {first_line}")
        else:
            skip_count += 1
            print(f"  Cell {i:3d}: SKIP   {first_line}")

print(f"\n{'='*60}")
print(f"SUMMARY: {ok_count} OK, {len(errors)} errors, {skip_count} skipped")
if errors:
    print("\nERRORS:")
    for cell_i, first_line, err in errors:
        print(f"  Cell {cell_i}: {err[:200]}")
else:
    print("ALL CELLS PASSED!")
