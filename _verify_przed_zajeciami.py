#!/usr/bin/env python3
"""
Weryfikacja: symuluje przed_zajeciami.py na executed test notebook,
porównuje znak-po-znaku z oryginalnym Function_Calling.ipynb.

UWAGA: _test_notebook.py zamienia cell 5 na bezpośredni connect do portu 4242.
       Dlatego po przed_zajeciami cell 5 NIE wróci do oryginału.
       Ten skrypt pomija cell 5 w porównaniu (znana różnica).
"""
import json
import copy
import re
import sys

ORIGINAL = "Function_Calling.ipynb"
TEST_EXECUTED = "Function_Calling_TEST_executed.ipynb"

# 1. Wczytaj oryginał
with open(ORIGINAL, encoding="utf-8") as f:
    orig = json.load(f)

# 2. Wczytaj executed test notebook
with open(TEST_EXECUTED, encoding="utf-8") as f:
    test = json.load(f)

# 3. Symuluj PEŁNĄ logikę przed_zajeciami.py na test notebook
print("Symulacja przed_zajeciami.py na test notebook...")

for i, cell in enumerate(test["cells"]):
    # ── 1. Reset komórek student-stub ──
    if cell["cell_type"] == "code":
        tags = cell.get("metadata", {}).get("tags", [])
        if "student-stub" in tags:
            template = cell.get("metadata", {}).get("student_stub_template")
            if template is not None:
                cell["source"] = list(template)
                cell["outputs"] = []
                cell["execution_count"] = None
                print(f"  Cell {i}: stub reset")
                continue

    # ── 2. Wyczyść outputy i licznik ──
    if cell["cell_type"] == "code":
        if cell.get("outputs") or cell.get("execution_count"):
            cell["outputs"] = []
            cell["execution_count"] = None

    # ── 3. tensorboard (not in FC notebook) ──
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        if "tensorboard --logdir=" in src:
            m = re.search(r"tensorboard --logdir=(\S+)", src)
            if m:
                logdir = m.group(1)
                canonical = [
                    "## Uruchom w terminalu (lub odkomentuj i uruchom tutaj):\n",
                    f"# !tensorboard --logdir={logdir}",
                ]
                cell["source"] = canonical

    # ── 4. Zwiń Rozwiązanie / Podpowiedź ──
    if cell["cell_type"] == "markdown":
        src = "".join(cell.get("source", []))
        first_line = src.lstrip().split("\n")[0]
        is_h6 = first_line.startswith("######")
        has_keyword = "Rozwiązanie" in first_line or "Podpowiedź" in first_line or "Spodziewany wynik" in first_line
        if is_h6 and has_keyword:
            cell.setdefault("metadata", {})["jp-MarkdownHeadingCollapsed"] = True

    # ── 5. Zakomentuj model= w connect_llm ──
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        if "connect_llm" in src:
            new_lines = []
            changed = False
            for line in cell["source"]:
                stripped = line.lstrip()
                if re.match(r'(model|backend)\s*=\s*"', stripped) and not stripped.startswith("#"):
                    indent = line[: len(line) - len(stripped)]
                    new_lines.append(f"{indent}# {stripped}")
                    changed = True
                else:
                    new_lines.append(line)
            if changed:
                cell["source"] = new_lines

    # ── 6. Reset API_KEY ──
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        if "API_KEY" in src:
            new_lines = []
            for line in cell["source"]:
                stripped = line.lstrip()
                if re.match(r'API_KEY\s*=\s*(?!None)', stripped):
                    indent = line[: len(line) - len(stripped)]
                    new_lines.append(f'{indent}API_KEY = None  # ← wpisz klucz API jeśli serwer wymaga autentykacji (prowadzący poda)\n')
                else:
                    new_lines.append(line)
            cell["source"] = new_lines


# 4. Porównaj z oryginałem — komórka po komórce
print(f"\n{'='*60}")
print("PORÓWNANIE Z ORYGINAŁEM:")
print(f"  Oryginał: {len(orig['cells'])} komórek")
print(f"  Test:     {len(test['cells'])} komórek")

if len(orig["cells"]) != len(test["cells"]):
    print(f"  RÓŻNA LICZBA KOMÓREK!")
    sys.exit(1)

diffs = []
skipped = []

for i, (o_cell, t_cell) in enumerate(zip(orig["cells"], test["cells"])):
    # Cell 5 was replaced by _test_notebook.py — known difference, skip
    if i == 5:
        skipped.append(i)
        continue

    # Compare source
    o_src = ''.join(o_cell.get("source", []))
    t_src = ''.join(t_cell.get("source", []))

    if o_src != t_src:
        diffs.append(("source", i, o_src, t_src))

    # Compare cell_type
    if o_cell.get("cell_type") != t_cell.get("cell_type"):
        diffs.append(("cell_type", i, o_cell.get("cell_type"), t_cell.get("cell_type")))

    # Compare metadata tags
    o_tags = o_cell.get("metadata", {}).get("tags", [])
    t_tags = t_cell.get("metadata", {}).get("tags", [])
    if o_tags != t_tags:
        diffs.append(("tags", i, o_tags, t_tags))

    # Compare student_stub_template
    o_tmpl = o_cell.get("metadata", {}).get("student_stub_template")
    t_tmpl = t_cell.get("metadata", {}).get("student_stub_template")
    if o_tmpl != t_tmpl:
        diffs.append(("template", i,
                      f"{len(o_tmpl)} lines" if o_tmpl else "None",
                      f"{len(t_tmpl)} lines" if t_tmpl else "None"))

    # Compare outputs (should both be empty)
    o_out = o_cell.get("outputs", [])
    t_out = t_cell.get("outputs", [])
    if o_out != t_out:
        diffs.append(("outputs", i, f"{len(o_out)} outputs", f"{len(t_out)} outputs"))

    # Compare execution_count
    o_ec = o_cell.get("execution_count")
    t_ec = t_cell.get("execution_count")
    if o_ec != t_ec:
        diffs.append(("execution_count", i, o_ec, t_ec))

    # Compare jp-MarkdownHeadingCollapsed
    o_collapsed = o_cell.get("metadata", {}).get("jp-MarkdownHeadingCollapsed")
    t_collapsed = t_cell.get("metadata", {}).get("jp-MarkdownHeadingCollapsed")
    if o_collapsed != t_collapsed:
        diffs.append(("collapsed", i, o_collapsed, t_collapsed))

print(f"\n  Sprawdzono {len(orig['cells'])} komórek (pominięto: {skipped})")

if not diffs:
    print("  IDENTYCZNE! Notebook wrócił do stanu wyjściowego!")
    print("  (pomijając cell 5 — zmieniony przez _test_notebook.py)")
else:
    print(f"  ZNALEZIONO {len(diffs)} RÓŻNIC:")
    for kind, cell_i, orig_val, test_val in diffs:
        print(f"\n  Cell {cell_i} [{kind}]:")
        if kind == "source":
            o_lines = orig_val.split("\n")
            t_lines = test_val.split("\n")
            print(f"    orig: {len(o_lines)} lines, test: {len(t_lines)} lines")
            for j, (ol, tl) in enumerate(zip(o_lines, t_lines)):
                if ol != tl:
                    print(f"    First diff at line {j}:")
                    print(f"      orig: {repr(ol[:120])}")
                    print(f"      test: {repr(tl[:120])}")
                    break
            if len(o_lines) != len(t_lines):
                print(f"    Line count: orig={len(o_lines)} vs test={len(t_lines)}")
                # Show extra lines
                if len(t_lines) > len(o_lines):
                    print(f"    Extra test lines:")
                    for l in t_lines[len(o_lines):len(o_lines)+3]:
                        print(f"      {repr(l[:120])}")
                else:
                    print(f"    Extra orig lines:")
                    for l in o_lines[len(t_lines):len(t_lines)+3]:
                        print(f"      {repr(l[:120])}")
        else:
            print(f"    orig: {orig_val}")
            print(f"    test: {test_val}")

# Also check source-level array equality (not just joined string)
print(f"\n{'='*60}")
print("SOURCE ARRAY CHECK (list comparison, not string):")
array_diffs = 0
for i, (o_cell, t_cell) in enumerate(zip(orig["cells"], test["cells"])):
    if i == 5:
        continue
    o_src = o_cell.get("source", [])
    t_src = t_cell.get("source", [])
    if o_src != t_src:
        array_diffs += 1
        if array_diffs <= 5:
            print(f"  Cell {i}: source arrays differ")
            print(f"    orig: {len(o_src)} elements")
            print(f"    test: {len(t_src)} elements")
            for j, (ol, tl) in enumerate(zip(o_src, t_src)):
                if ol != tl:
                    print(f"    First diff at elem {j}: {repr(ol[:80])} vs {repr(tl[:80])}")
                    break
if array_diffs == 0:
    print("  All source arrays match!")
else:
    print(f"  {array_diffs} cells have different source arrays")
