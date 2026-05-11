#!/usr/bin/env python3
"""
Builder for Function_Calling.ipynb.

Czyta definicje komórek z _fc_cells.json i generuje notebook.
Aby zregenerować _fc_cells.json z istniejącego notebooka:
    python -c "
    import json
    nb = json.load(open('Function_Calling_final2.ipynb'))
    cells = []
    for c in nb['cells']:
        m = c.get('metadata', {})
        cells.append({
            'type': c['cell_type'],
            'stub': 'student_stub_template' in m,
            'collapsed': m.get('jp-MarkdownHeadingCollapsed', False),
            'source': ''.join(c['source']),
        })
    json.dump(cells, open('_fc_cells.json','w'), ensure_ascii=False, indent=1)
    "

Użycie:
    python _build_fc_notebook.py                     # → Function_Calling.ipynb
    python _build_fc_notebook.py output.ipynb         # → output.ipynb
"""

import json
import sys
from pathlib import Path

OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "Function_Calling.ipynb"
CELLS_JSON = Path(__file__).parent / "_fc_cells.json"


# ── Helpers ─────────────────────────────────────────────────────────

def _split(source):
    """Split source into JSON-compatible line list."""
    lines = source.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(source, collapsed=False):
    """Markdown cell."""
    meta = {}
    if collapsed:
        meta["jp-MarkdownHeadingCollapsed"] = True
    return {"cell_type": "markdown", "metadata": meta, "source": _split(source)}


def code(source):
    """Code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split(source),
    }


def student_stub(source):
    """Code cell that przed_zajeciami.py resets to its template."""
    cell = code(source)
    cell["metadata"]["student_stub_template"] = _split(source)
    return cell


# ── Build ───────────────────────────────────────────────────────────

with open(CELLS_JSON, encoding="utf-8") as f:
    cell_data = json.load(f)

cells = []
for entry in cell_data:
    src = entry["source"]
    if entry["type"] == "markdown":
        cells.append(md(src, collapsed=entry.get("collapsed", False)))
    elif entry.get("stub"):
        cells.append(student_stub(src))
    else:
        cells.append(code(src))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (ml)-UV",
            "language": "python",
            "name": "ml",
        },
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Zapisano {OUTPUT} ({len(cells)} komórek)")
