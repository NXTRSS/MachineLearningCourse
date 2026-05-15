# Plan A — uv ⭐ (polecany)

`uv` to szybki i lekki menedżer środowisk Pythona. Działa na Windowsie, Macu i Linuksie. Nie wymaga Condy, Anacondy, Dockera ani żadnych wirtualnych maszyn.


## Krok 1: Instalacja uv

### Windows (PowerShell)

Otwórz PowerShell i wpisz:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Po instalacji zamknij i otwórz ponownie PowerShell.

### macOS / Linux

Otwórz terminal i wpisz:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Po instalacji zamknij i otwórz ponownie terminal.

### Sprawdzenie instalacji

```bash
uv --version
```

Powinno wyświetlić coś w stylu: `uv 0.6.x`


## Krok 2: Instalacja Git

Sprawdź czy masz Gita:

```bash
git --version
```

Jeśli komenda zwraca wersję (np. `git version 2.x`) — przejdź do **Kroku 3**.

Jeśli nie masz Gita:

### Windows (PowerShell)

```powershell
winget install Git.Git
```

> Pojawi się okno UAC (kontrola konta użytkownika) — kliknij **Tak**. Po instalacji zamknij i otwórz ponownie PowerShell.

### macOS

```bash
xcode-select --install
```

Pojawi się okno dialogowe — kliknij **Install**. Instalacja trwa 1-2 minuty.

> Alternatywa jeśli masz Homebrew: `brew install git` (nie wymaga uprawnień administratora).

### Linux (Ubuntu / Debian)

```bash
sudo apt-get install git
```

> Wymaga hasła administratora (`sudo`).


## Krok 3: Pobranie plików z zajęć

```bash
git clone https://github.com/NXTRSS/MachineLearningCourse
cd MachineLearningCourse
```

> **Nie możesz zainstalować Gita?** Alternatywnie:
> 1. Wejdź na https://github.com/NXTRSS/MachineLearningCourse
> 2. Kliknij zielony przycisk **Code** → **Download ZIP**
> 3. Wypakuj archiwum i otwórz terminal w tym folderze
>
> ⚠️ Bez Gita nie będziesz mógł pobierać aktualizacji materiałów (`git pull`).


## Krok 4: Stworzenie środowiska i instalacja pakietów

W folderze `MachineLearningCourse` wpisz:

```bash
uv sync
```

To **jedyna komenda** — `uv` automatycznie:
- Pobierze i zainstaluje Python 3.9 (jeśli go nie masz)
- Stworzy wirtualne środowisko (`.venv`)
- Zainstaluje wszystkie wymagane pakiety (TensorFlow, Pandas, JupyterLab, itd.)

Może to potrwać kilka minut przy pierwszym uruchomieniu.


## Krok 5: Instalacja kernela Jupyter

```bash
uv run python -m ipykernel install --user --name ml --display-name "Python (ml)"
```


## Krok 6: Uruchomienie JupyterLab

```bash
uv run jupyter lab
```

Otworzy się przeglądarka z JupyterLab. Gotowe!

**Ważne:** W JupyterLab jako kernel wybierz **"Python (ml)"**.


## Krok 7: Weryfikacja

W JupyterLab otwórz Terminal (File → New → Terminal) i wpisz:

```bash
uv run python verify_env.py
```


## Rozpoczęcie pracy na kolejny dzień

Przy każdym kolejnym uruchomieniu wystarczy:

1. Otworzyć terminal w folderze `MachineLearningCourse`
2. Wpisać:
   ```bash
   uv run jupyter lab
   ```

Nie trzeba niczego aktywować — `uv run` automatycznie używa odpowiedniego środowiska.


## Instalacja graphviz (wymagana do wizualizacji modeli)

`graphviz` to program systemowy (nie pythonowa paczka) — trzeba go zainstalować osobno:

### Windows
Pobierz instalator z: https://graphviz.org/download/ (sekcja Windows) i zainstaluj. Podczas instalacji zaznacz **"Add Graphviz to the system PATH"**.

### macOS
```bash
brew install graphviz
```
Jeśli nie masz `brew`: https://brew.sh

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install graphviz
```
> Wymaga hasła administratora (`sudo`).


## Rozwiązywanie problemów

### "uv: command not found"
Zamknij i otwórz ponownie terminal po instalacji uv. Jeśli dalej nie działa, dodaj uv do PATH ręcznie — instalator powinien był wypisać ścieżkę.

### TensorFlow nie instaluje się na M1/M2/M3 Mac
TensorFlow 2.16+ ma pełne wsparcie dla Apple Silicon. Jeśli są problemy, spróbuj:
```bash
uv sync --reinstall
```

### "No module named..." przy uruchomieniu notebooka
Upewnij się, że w JupyterLab wybrany jest kernel **"Python (ml)"**, a nie domyślny "Python 3".


## Nie działa? Przejdź do Planu C

**➡️ [Plan C — Google Colab](PLAN_C_COLAB.md)**
