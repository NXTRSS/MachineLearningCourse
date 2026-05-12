"""
przed_zajeciami.py
==================
Uruchom przed każdymi zajęciami:
    python przed_zajeciami.py

Co robi:
  1. Resetuje komórki oznaczone jako student-stub (przywraca ich kanoniczną treść
     z metadanych — czyści ślady live codingu z poprzedniej grupy)
  2. Czyści wszystkie outputy i execution_count (In [1]:, wyniki, wykresy)
  3. Zwija wszystkie komórki Podpowiedź / Rozwiązanie
  4. Zakomentowuje komórki z !tensorboard (żeby student nie odpalił przypadkiem)
  5. Zakomentowuje backend=/ports= w connect_llm (żeby auto-detect działał)
  6. Resetuje API_KEY do None (żeby klucz prowadzącego nie wyciekł do repo)
  7. Resetuje LECTURER_SERVER do placeholdera (żeby IP prowadzącego nie wyciekł)
  8. Zapisuje — notebook gotowy dla studentów

Patrz: STUBY_RESETOWANIE.md — jak oznaczać komórki jako student-stub.
"""

import json
import re
from datetime import datetime
from pathlib import Path

NOTEBOOKI = [
    "Regresja_liniowa_rozszerzona.ipynb",
    "Regresja_liniowa.ipynb",
    "Regresja_logistyczna.ipynb",
    "Reprezentacja_tekstu.ipynb",
    "Sieć_Neuronowa.ipynb",
    "Model_Języka_polskie_nazwy_miast.ipynb",
    "Function_Calling.ipynb",
]


def wyczysc_i_przygotuj(path: Path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    stuby_zresetowane = 0
    stuby_bez_szablonu = 0
    komórki_wyczyszczone = 0
    komórki_zwinięte = 0
    tb_zakomentowane = 0
    model_zakomentowane = 0
    api_key_zresetowane = 0
    lecturer_zresetowane = 0

    for cell in nb["cells"]:
        # ── 1. Reset komórek student-stub ────────────────────────────────────
        # Komórki oznaczone tagiem "student-stub" mają w metadanych pole
        # "student_stub_template" — kanoniczna treść do której wracamy.
        if cell["cell_type"] == "code":
            tags = cell.get("metadata", {}).get("tags", [])
            if "student-stub" in tags:
                template = cell.get("metadata", {}).get("student_stub_template")
                if template is None:
                    stuby_bez_szablonu += 1
                else:
                    cell["source"] = list(template)
                    cell["outputs"] = []
                    cell["execution_count"] = None
                    stuby_zresetowane += 1
                    continue  # już wyczyszczone — nie dubluj w pkt. 2

        # ── 2. Wyczyść outputy i licznik wykonania ──────────────────────────
        if cell["cell_type"] == "code":
            if cell.get("outputs") or cell.get("execution_count"):
                cell["outputs"] = []
                cell["execution_count"] = None
                komórki_wyczyszczone += 1

        # ── 3. Resetuj komórki z !tensorboard do kanonicznej formy ────────
        # Student mógł odkomentować (Ctrl+/) → ## stało się #, a komenda
        # się odpaliła. Przywracamy pełną zakomentowaną wersję.
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
                    if cell["source"] != canonical:
                        cell["source"] = canonical
                        tb_zakomentowane += 1

        # ── 4. Zwiń komórki Rozwiązanie / Podpowiedź / Spodziewany wynik ──
        # Zwijamy tylko komórki ZACZYNAJĄCE SIĘ od ###### (h6) z słowem
        # kluczowym — nie dotykamy komórek które tylko wspominają te słowa
        # w treści (np. opis legendy na początku notebooka).
        if cell["cell_type"] == "markdown":
            src = "".join(cell.get("source", []))
            first_line = src.lstrip().split("\n")[0]
            is_h6 = first_line.startswith("######")
            has_keyword = "Rozwiązanie" in first_line or "Podpowiedź" in first_line or "Spodziewany wynik" in first_line
            if is_h6 and has_keyword:
                cell.setdefault("metadata", {})["jp-MarkdownHeadingCollapsed"] = True
                komórki_zwinięte += 1

        # ── 4b. Zwiń sekcje dodatkowe (## Xb., ## Xc. itd.) ────────────────
        # Podsekcje oznaczone literą (10b., 8b.) zwijamy domyślnie —
        # to materiał dodatkowy, student może rozwinąć jeśli chce.
        if cell["cell_type"] == "markdown":
            src = "".join(cell.get("source", []))
            first_line = src.lstrip().split("\n")[0]
            if re.match(r"^##\s+\d+[b-z]\.", first_line):
                cell.setdefault("metadata", {})["jp-MarkdownHeadingCollapsed"] = True
                komórki_zwinięte += 1

        # ── 5. Zakomentuj backend=/ports= w connect_llm ─────────────────────
        # Prowadzący może odkomentować na swoim laptopie.
        # Przed pushem do studentów zakomentowujemy, żeby auto-detect działał.
        # model= zostawiamy odkomentowane (gemma-4-e4b to domyślny model na zajęciach).
        if cell["cell_type"] == "code":
            src = "".join(cell.get("source", []))
            if "connect_llm" in src:
                new_lines = []
                changed = False
                for line in cell["source"]:
                    stripped = line.lstrip()
                    if re.match(r'(backend|ports)\s*=', stripped) and not stripped.startswith("#"):
                        indent = line[: len(line) - len(stripped)]
                        new_lines.append(f"{indent}# {stripped}")
                        changed = True
                    else:
                        new_lines.append(line)
                if changed:
                    cell["source"] = new_lines
                    model_zakomentowane += 1

        # ── 6. Reset API_KEY do None ────────────────────────────────────────
        # Prowadzący wpisuje klucz na zajęciach; przed pushem zerujemy,
        # żeby nie wyciekł do repozytorium.
        if cell["cell_type"] == "code":
            src = "".join(cell.get("source", []))
            if "API_KEY" in src:
                new_lines = []
                changed = False
                for line in cell["source"]:
                    stripped = line.lstrip()
                    if re.match(r'API_KEY\s*=\s*(?!None)', stripped):
                        indent = line[: len(line) - len(stripped)]
                        new_lines.append(f'{indent}API_KEY = None  # ← wpisz klucz API jeśli serwer wymaga autentykacji (prowadzący poda)\n')
                        changed = True
                    else:
                        new_lines.append(line)
                if changed:
                    cell["source"] = new_lines
                    api_key_zresetowane += 1

        # ── 7. Reset LECTURER_SERVER do placeholdera ───────────────────────
        # Prowadzący wpisuje swój IP na zajęciach; przed pushem zerujemy.
        if cell["cell_type"] == "code":
            src = "".join(cell.get("source", []))
            if "LECTURER_SERVER" in src:
                new_lines = []
                changed = False
                for line in cell["source"]:
                    stripped = line.lstrip()
                    if re.match(r'LECTURER_SERVER\s*=\s*', stripped) and "ADRES_SERWERA" not in stripped:
                        indent = line[: len(line) - len(stripped)]
                        new_lines.append(f'{indent}LECTURER_SERVER = "http://ADRES_SERWERA:PORT"  # ← prowadzący poda na zajęciach\n')
                        changed = True
                    else:
                        new_lines.append(line)
                if changed:
                    cell["source"] = new_lines
                    lecturer_zresetowane += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"  ✓ {path.name}")
    print(f"    — zresetowano {stuby_zresetowane} komórek student-stub")
    if stuby_bez_szablonu:
        print(f"    ⚠ {stuby_bez_szablonu} komórek z tagiem student-stub bez szablonu — uzupełnij przez oznacz_stub.py")
    if stuby_zresetowane == 0 and stuby_bez_szablonu == 0:
        print(f"      (notebook nie ma oznaczonych stubów — odpal: python oznacz_stub.py {path.name} <indeksy>)")
    print(f"    — wyczyszczono {komórki_wyczyszczone} komórek z outputami")
    print(f"    — zwinięto {komórki_zwinięte} komórek Podpowiedź/Rozwiązanie")
    if tb_zakomentowane:
        print(f"    — zakomentowano {tb_zakomentowane} komórek !tensorboard")
    if model_zakomentowane:
        print(f"    — zakomentowano {model_zakomentowane} linii model= w connect_llm")
    if api_key_zresetowane:
        print(f"    — zresetowano {api_key_zresetowane} linii API_KEY do None")
    if lecturer_zresetowane:
        print(f"    — zresetowano {lecturer_zresetowane} linii LECTURER_SERVER do placeholdera")


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
