"""
przed_zajeciami.py
==================
Uruchom przed każdymi zajęciami:
    python przed_zajeciami.py

Co robi:
  1. Czyści wszystkie outputy i execution_count (In [1]:, wyniki, wykresy)
  2. Zwija wszystkie komórki Podpowiedź / Rozwiązanie
  3. Zapisuje — notebook gotowy dla studentów
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

NOTEBOOKI = [
    "Regresja_liniowa_rozszerzona.ipynb",
    "Regresja_liniowa.ipynb",
]

def wyczysc_i_przygotuj(path: Path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    komórki_wyczyszczone = 0
    komórki_zwinięte = 0

    for cell in nb["cells"]:
        # ── 1. Wyczyść outputy i licznik wykonania ──────────────────────────
        if cell["cell_type"] == "code":
            if cell.get("outputs") or cell.get("execution_count"):
                cell["outputs"] = []
                cell["execution_count"] = None
                komórki_wyczyszczone += 1

        # ── 2. Zwiń komórki zwijalne ─────────────────────────────────────────
        # Tylko komórki, które już MAJĄ klucz jp-MarkdownHeadingCollapsed w
        # metadanych — tzn. zostały celowo zaprojektowane jako zwijalne.
        # Nie dodajemy tego klucza do nowych komórek (nie zgadujemy po treści).
        if "jp-MarkdownHeadingCollapsed" in cell.get("metadata", {}):
            if not cell["metadata"]["jp-MarkdownHeadingCollapsed"]:
                cell["metadata"]["jp-MarkdownHeadingCollapsed"] = True
                komórki_zwinięte += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"  ✓ {path.name}")
    print(f"    — wyczyszczono {komórki_wyczyszczone} komórek z outputami")
    print(f"    — zwinięto {komórki_zwinięte} komórek Podpowiedź/Rozwiązanie")


def main():
    repo = Path(__file__).parent
    print(f"\n{'='*55}")
    print(f"  Przygotowanie notebooków — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'='*55}\n")

    for nazwa in NOTEBOOKI:
        path = repo / nazwa
        if not path.exists():
            print(f"  ⚠  Pominięto (nie znaleziono): {nazwa}")
            continue
        wyczysc_i_przygotuj(path)
        print()

    print("Gotowe — notebooki przygotowane dla studentów.\n")


if __name__ == "__main__":
    main()
