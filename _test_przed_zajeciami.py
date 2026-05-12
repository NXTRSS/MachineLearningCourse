"""
Test E2E dla przed_zajeciami.py + Function_Calling.ipynb

Uruchomienie:
    python _test_przed_zajeciami.py

Co sprawdza:
  1. Czy wszystkie 6 stubów ma tag student-stub + szablon
  2. Czy szablony zgadzają się z final4_chat_pass (źródło prawdy)
  3. Dirty → przed_zajeciami → czy stuby wracają do czysta
  4. Czy connect_llm cell jest poprawnie resetowany (LECTURER_SERVER, backend, ports, model)
  5. Czy nie ma surowych bajtów ESC ( / \x1b) w pliku
  6. Czy outputs i execution_count są wyczyszczone
"""

import json
import copy
import subprocess
import sys
from pathlib import Path

FC_PATH = Path("Function_Calling.ipynb")
FP_PATH = Path("Function_Calling_final4_chat_pass.ipynb")

# Indeksy stubów w obu notebookach
STUB_CELLS = {
    40: 40,   # Ćw. 1A
    55: 55,   # Ćw. 2
    64: 64,   # Ćw. 3
    81: 81,   # Ćw. 4
    97: 97,   # Ćw. 5
    102: 102, # Ćw. 6
}

CONNECT_LLM_CELL = 5

errors = []
warnings = []


def err(msg):
    errors.append(msg)
    print(f"  ❌ {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"  ⚠️  {msg}")


def ok(msg):
    print(f"  ✅ {msg}")


def src(cell):
    return "".join(cell.get("source", []))


# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  TEST: przed_zajeciami.py + Function_Calling.ipynb")
print("=" * 60)

fc = json.loads(FC_PATH.read_text(encoding="utf-8"))
fp = json.loads(FP_PATH.read_text(encoding="utf-8"))

# ── 1. Sprawdź tagi i szablony ──────────────────────────────
print("\n1. Tagi student-stub i szablony:")
for fc_idx in STUB_CELLS:
    cell = fc["cells"][fc_idx]
    tags = cell.get("metadata", {}).get("tags", [])
    tpl = cell.get("metadata", {}).get("student_stub_template")
    label = src(cell).split("\n")[0][:50]

    if "student-stub" not in tags:
        err(f"Cell {fc_idx}: BRAK tagu student-stub — {label}")
    elif tpl is None:
        err(f"Cell {fc_idx}: BRAK student_stub_template — {label}")
    elif len(tpl) == 0:
        err(f"Cell {fc_idx}: PUSTY student_stub_template — {label}")
    else:
        ok(f"Cell {fc_idx}: tag + szablon OK — {label}")

# ── 2. Porównaj szablony z final4_chat_pass ──────────────────
print("\n2. Szablony vs final4_chat_pass (źródło prawdy):")
for fc_idx, fp_idx in STUB_CELLS.items():
    tpl = fc["cells"][fc_idx].get("metadata", {}).get("student_stub_template", [])
    tpl_src = "".join(tpl) if isinstance(tpl, list) else (tpl or "")
    fp_src = src(fp["cells"][fp_idx])

    if tpl_src == fp_src:
        ok(f"Cell {fc_idx}: szablon = final4_chat_pass ✓")
    else:
        err(f"Cell {fc_idx}: szablon RÓŻNI SIĘ od final4_chat_pass!")
        # Show first difference
        tpl_lines = tpl_src.splitlines()
        fp_lines = fp_src.splitlines()
        for i, (a, b) in enumerate(zip(tpl_lines, fp_lines)):
            if a != b:
                print(f"         Linia {i+1} FC:  {a[:80]}")
                print(f"         Linia {i+1} FP:  {b[:80]}")
                break
        if len(tpl_lines) != len(fp_lines):
            print(f"         FC: {len(tpl_lines)} linii vs FP: {len(fp_lines)} linii")

# ── 3. Porównaj BIEŻĄCĄ treść stubów z szablonami ───────────
print("\n3. Bieżąca treść stubów = szablony (po ostatnim resecie):")
for fc_idx in STUB_CELLS:
    cell = fc["cells"][fc_idx]
    current = src(cell)
    tpl = cell.get("metadata", {}).get("student_stub_template", [])
    tpl_src = "".join(tpl) if isinstance(tpl, list) else (tpl or "")
    if current == tpl_src:
        ok(f"Cell {fc_idx}: source = template ✓")
    else:
        warn(f"Cell {fc_idx}: source ≠ template (drift) — może OK jeśli przed_zajeciami jeszcze nie był uruchomiony")

# ── 4. Dirty → przed_zajeciami → sprawdzenie ────────────────
print("\n4. Test E2E: dirty → przed_zajeciami.py → weryfikacja:")

# Backup
backup = json.dumps(fc, ensure_ascii=False)

# Dirty up
dirty = json.loads(backup)
for fc_idx in STUB_CELLS:
    dirty["cells"][fc_idx]["source"] = [f"# DIRTY cell {fc_idx}\npass\n"]
    dirty["cells"][fc_idx]["execution_count"] = 99
    dirty["cells"][fc_idx]["outputs"] = [{"output_type": "stream", "text": ["fake\n"]}]

# Dirty cell 5
cell5 = dirty["cells"][CONNECT_LLM_CELL]
new_src = []
for line in cell5["source"]:
    s = line.lstrip()
    if s.startswith("# backend=") and not s.startswith("# backend=["):
        indent = line[: len(line) - len(s)]
        new_src.append(indent + s[2:])
    elif s.startswith("# ports="):
        indent = line[: len(line) - len(s)]
        new_src.append(indent + s[2:])
    elif "ADRES_SERWERA" in s and "LECTURER_SERVER" in s:
        indent = line[: len(line) - len(s)]
        new_src.append(f'{indent}LECTURER_SERVER = "http://10.0.0.99:4242"\n')
    else:
        new_src.append(line)
cell5["source"] = new_src
cell5["execution_count"] = 99

# Save dirty version
FC_PATH.write_text(json.dumps(dirty, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

# Run przed_zajeciami.py
result = subprocess.run(
    [sys.executable, "przed_zajeciami.py"],
    capture_output=True, text=True
)
print(f"  przed_zajeciami.py stdout:")
for line in result.stdout.strip().split("\n"):
    if "Function_Calling" in line or line.strip().startswith("—") or line.strip().startswith("⚠"):
        print(f"    {line.strip()}")

# Load result
cleaned = json.loads(FC_PATH.read_text(encoding="utf-8"))

# Check stubs
stubs_ok = 0
for fc_idx, fp_idx in STUB_CELLS.items():
    cleaned_src = src(cleaned["cells"][fc_idx])
    fp_source = src(fp["cells"][fp_idx])
    ec = cleaned["cells"][fc_idx].get("execution_count")
    outs = len(cleaned["cells"][fc_idx].get("outputs", []))

    if cleaned_src != fp_source:
        err(f"Cell {fc_idx}: po resecie ≠ final4_chat_pass!")
    elif ec is not None:
        err(f"Cell {fc_idx}: execution_count={ec} (powinno None)")
    elif outs > 0:
        err(f"Cell {fc_idx}: {outs} outputs (powinno 0)")
    else:
        stubs_ok += 1

if stubs_ok == len(STUB_CELLS):
    ok(f"Wszystkie {stubs_ok} stubów zresetowane poprawnie")
else:
    err(f"Tylko {stubs_ok}/{len(STUB_CELLS)} stubów OK")

# Check cell 5
cell5_src = src(cleaned["cells"][CONNECT_LLM_CELL])
cell5_lines = cell5_src.split("\n")

checks = {
    "LECTURER_SERVER placeholder": any("ADRES_SERWERA" in l for l in cell5_lines),
    "model= odkomentowane": any(l.strip().startswith("model=") for l in cell5_lines),
    "backend= zakomentowane": not any(
        l.strip().startswith("backend=") for l in cell5_lines
    ),
    "ports= zakomentowane": not any(
        l.strip().startswith("ports=") for l in cell5_lines
    ),
    "execution_count=None": cleaned["cells"][CONNECT_LLM_CELL].get("execution_count") is None,
}
for label, passed in checks.items():
    if passed:
        ok(f"Cell 5: {label}")
    else:
        err(f"Cell 5: {label} — FAILED")

# Restore original
FC_PATH.write_text(json.dumps(fc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

# ── 5. Sprawdź surowe bajty ESC ─────────────────────────────
print("\n5. Surowe bajty ESC w pliku:")
raw = FC_PATH.read_bytes()
esc_count = raw.count(b"\x1b")
u001b_count = raw.count(b"\\u001b")
if esc_count > 0:
    err(f"Znaleziono {esc_count} surowych bajtów ESC (0x1b)")
elif u001b_count > 0:
    err(f"Znaleziono {u001b_count} sekwencji \\u001b (powinno być \\033)")
else:
    ok("Brak surowych ESC / \\u001b — czysto")

# ── 6. Sprawdź czy WSZYSTKIE outputs wyczyszczone ───────────
print("\n6. Outputs po resecie:")
dirty_outputs = 0
for idx, cell in enumerate(cleaned["cells"]):
    if cell.get("cell_type") == "code":
        if cell.get("outputs") or cell.get("execution_count") is not None:
            dirty_outputs += 1
if dirty_outputs == 0:
    ok("Wszystkie komórki code: outputs=[], execution_count=None")
else:
    warn(f"{dirty_outputs} komórek nadal ma outputs/execution_count")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
if errors:
    print(f"  ❌ FAILED — {len(errors)} błędów, {len(warnings)} ostrzeżeń")
    for e in errors:
        print(f"     • {e}")
else:
    print(f"  ✅ ALL PASSED — 0 błędów, {len(warnings)} ostrzeżeń")
print("=" * 60)

sys.exit(1 if errors else 0)
