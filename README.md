# Zajęcia z Uczenia Maszynowego

## Który plan wybrać?

| Plan | Dla kogo? | Wymaga |
|------|-----------|--------|
| **[Plan A — uv](docs/PLAN_A_UV.md)** ⭐ *polecany* | Dla wszystkich | uv (~50 MB) |
| **[Plan B — Docker](docs/PLAN_B_DOCKER.md)** | Nie chcę konfigurować Pythona | Docker Desktop |
| **[Plan C — Google Colab](docs/PLAN_C_COLAB.md)** | Ostateczność — nic innego nie działa | Tylko przeglądarka |

> **Dlaczego uv?** Jednorazowa konfiguracja (~5 min), startuje natychmiast, działa natywnie na Windows / Mac / Linux bez VM. Uczy też umiejętności przydatnej poza zajęciami.

---

## 🔧 Pierwsze uruchomienie (robimy raz)

### Plan A — uv

```bash
# 1. Zainstaluj uv (jednorazowo)
curl -LsSf https://astral.sh/uv/install.sh | sh    # macOS / Linux
winget install astral-sh.uv                         # Windows (PowerShell)

# 2. Sklonuj repo
git clone https://github.com/NXTRSS/MachineLearningCourse
cd MachineLearningCourse

# 3. Zainstaluj środowisko i kernel
uv sync
uv run python -m ipykernel install --user --name ml --display-name "Python (ml)"

# 4. Uruchom JupyterLab
uv run jupyter lab
```

Pełna instrukcja + rozwiązywanie problemów: [docs/PLAN_A_UV.md](docs/PLAN_A_UV.md)

---

### Plan B — Docker

```bash
# 1. Zainstaluj Docker Desktop → https://docs.docker.com/desktop/
#    Uruchom go i poczekaj aż ikona przestanie się kręcić

# 2. Sklonuj repo
git clone https://github.com/NXTRSS/MachineLearningCourse
cd MachineLearningCourse

# 3. Uruchom (pierwsze pobranie obrazu: 5–15 min)
docker compose up
```

Pełna instrukcja + rozwiązywanie problemów: [docs/PLAN_B_DOCKER.md](docs/PLAN_B_DOCKER.md)

---

### Plan C — Google Colab

Wejdź na [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub** → wklej:
```
https://github.com/NXTRSS/MachineLearningCourse
```

Pełna instrukcja: [docs/PLAN_C_COLAB.md](docs/PLAN_C_COLAB.md)

---

## ▶️ Każde kolejne zajęcia (codzienny start)

### Plan A — uv

```bash
cd MachineLearningCourse
uv run jupyter lab
```

Otwórz przeglądarkę: **http://localhost:8888**

W JupyterLab wybierz kernel **Python (ml)** — nie domyślny Python 3.

---

### Plan B — Docker

1. Upewnij się, że **Docker Desktop jest uruchomiony** (ikona w zasobniku systemowym)
2. W terminalu:
```bash
cd MachineLearningCourse
docker compose up
```
3. Otwórz przeglądarkę: **http://localhost:8888**

Aby zatrzymać: `Ctrl+C` w terminalu lub `docker compose down`.

---

### Plan C — Google Colab

Wejdź na [colab.research.google.com](https://colab.research.google.com) i otwórz notebook z historii lub ponownie z GitHub.

⚠️ Colab nie zachowuje zainstalowanych paczek między sesjami — pierwsza komórka każdego notebooka instaluje je automatycznie.

---

## 🔄 Po aktualizacji materiałów

Przed zajęciami prowadzący może aktualizować notebooki. Żeby mieć najnowszą wersję:

**Plan A i B:**
```bash
cd MachineLearningCourse
git pull
uv sync          # tylko Plan A — jeśli zmieniły się zależności
```

> **`git pull` zgłasza błędy / konflikty?** Jeśli nie masz własnych zmian, które chcesz zachować:
> ```bash
> git fetch origin
> git reset --hard origin/main
> uv sync   # tylko Plan A
> ```
> To nadpisze lokalną kopię aktualną wersją z GitHub.

**Plan C:** pobierz notebook ponownie z GitHub (File → Open notebook → GitHub).

---

## ✅ Weryfikacja środowiska

Po pierwszej konfiguracji sprawdź czy wszystko działa:

**Plan A:**
```bash
uv run python verify_env.py
```

**Plan B** — w terminalu JupyterLab (File → New → Terminal):
```bash
python verify_env.py
```

Jeśli zobaczysz `Environment verification: OK` — gotowe!

---

## Wymagania sprzętowe

- **RAM**: minimum 6 GB (zalecane 8 GB)
- **Dysk**: ~3 GB (Plan A/B) lub brak (Plan C)
- **System**: Windows 10/11, macOS 12+, Linux (Ubuntu 20.04+)

---

## Ollama / LM Studio (warsztaty LLM)

Na zajęcia z Function Calling i RAG potrzebny jest lokalny model językowy.
Uruchom notebook **`setup_local_llm.ipynb`** — sprawdzi sprzęt i przeprowadzi przez instalację.

Chcesz przygotować się wcześniej? → **[docs/LOKALNE_LLM.md](docs/LOKALNE_LLM.md)**

Na zajęciach dostępny jest też serwer prowadzącego (adres podany na miejscu) → [docs/SERWER_PROWADZACEGO.md](docs/SERWER_PROWADZACEGO.md)

---

## Dane do zajęć

Pobierane automatycznie przy pierwszym uruchomieniu notebooka (przez `utils.py`).
Jeśli nie zadziała — link do Google Drive zostanie podany na zajęciach.

---

## Problemy?

- Przeczytaj komunikat błędu w terminalu i wklej go do ChatGPT / Claude / Gemini — to działa.
- Sprawdź szczegółowe instrukcje dla swojego planu w folderze [`docs/`](docs/).
- Na zajęciach zawsze można skorzystać z Google Colab jako backup.
