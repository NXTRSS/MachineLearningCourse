# Zajęcia z Uczenia Maszynowego

Witam na zajęciach! Poniżej znajdują się instrukcje przygotowania środowiska do pracy.

Przygotowałem **trzy opcje** — zaczynam od rekomendowanej, ale każda działa.


## Który plan wybrać?

| Plan | Dla kogo? | Co trzeba zainstalować? |
|------|-----------|------------------------|
| **[Plan A — uv](docs/PLAN_A_UV.md)** ⭐ *polecany* | Dla wszystkich — lekkie, szybkie, uczy zarządzania środowiskiem Pythona | uv (lekki menedżer Pythona) |
| **[Plan B — Docker](docs/PLAN_B_DOCKER.md)** | Nie chcę konfigurować Pythona lokalnie — wolę jedno kliknięcie | Docker Desktop |
| **[Plan C — Google Colab](docs/PLAN_C_COLAB.md)** | Ostateczność — gdy nic innego nie działa | Tylko przeglądarka |

> **Dlaczego uv?** Po jednorazowej konfiguracji (ok. 5 minut) środowisko startuje natychmiast, nie zużywa zbędnych zasobów i uczy umiejętności przydatnej daleko poza tymi zajęciami. Jeśli jednak konfigurowanie Pythona lokalnie brzmi zniechęcająco — Plan B z Dockerem też jest OK, zrobi wszystko za Ciebie.


## Szybki start (Plan A — uv)

```bash
# 1. Zainstaluj uv (jednorazowo)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# Windows: winget install astral-sh.uv

# 2. Sklonuj repo i uruchom
git clone https://github.com/NXTRSS/MachineLearningCourse
cd MachineLearningCourse
uv sync
uv run jupyter lab
```

Otwórz przeglądarkę: **http://localhost:8888** — gotowe!

Pełna instrukcja: [docs/PLAN_A_UV.md](docs/PLAN_A_UV.md)


## Wymagania sprzętowe

- **RAM**: minimum 6 GB (zalecane 8 GB)
- **Dysk**: ~3 GB wolnego miejsca
- **System**: Windows 10/11, macOS 12+, lub Linux (Ubuntu 20.04+)


## Weryfikacja środowiska

Po uruchomieniu środowiska (dowolnym planem) otwórz terminal w JupyterLab i uruchom:

```bash
python verify_env.py
```

Jeśli zobaczysz `Environment verification: OK` — wszystko jest gotowe!


## Ollama / LM Studio (warsztaty LLM)

Na zajęcia z Function Calling i RAG potrzebny jest lokalny model językowy.
Przed tymi zajęciami proszę uruchomić notebook **`setup_local_llm.ipynb`** — sprawdzi sprzęt i przeprowadzi przez instalację Ollamy lub LM Studio krok po kroku.

Chcesz przygotować się wcześniej w domu? Pełny przewodnik — porównanie platform, jakie modele pobrać, MLX na Apple Silicon, rozwiązywanie problemów:
**[docs/LOKALNE_LLM.md](docs/LOKALNE_LLM.md)**

Jeśli instalacja nie wyjdzie, na zajęciach można skorzystać z serwera prowadzącego (adres zostanie podany na miejscu).
Instrukcja podłączenia i rozwiązywanie problemów: [docs/SERWER_PROWADZACEGO.md](docs/SERWER_PROWADZACEGO.md)


## Dane do zajęć

Dane zostaną pobrane automatycznie przy pierwszym uruchomieniu odpowiednich notebooków (przez skrypt `utils.py`). Jeśli automatyczne pobieranie nie zadziała, dane będą dostępne na dysku Google — link zostanie udostępniony na zajęciach.


## Problemy?

Jeśli żaden plan nie działa, na zajęciach w ostateczności skorzystamy z Google Colab (Plan C).
