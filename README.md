# Zajęcia z Uczenia Maszynowego

Witam na zajęciach! Poniżej znajdują się instrukcje przygotowania środowiska do pracy.

Przygotowałem **trzy opcje** — proszę zacząć od Planu A. Jeśli Plan A zadziała, nie trzeba czytać dalej.


## Który plan wybrać?

| Plan | Dla kogo? | Co trzeba zainstalować? |
|------|-----------|------------------------|
| **[Plan A — Docker](docs/PLAN_A_DOCKER.md)** | Dla każdego (rekomendowany) | Docker Desktop |
| **[Plan B — uv](docs/PLAN_B_UV.md)** | Gdy Docker nie działa / nie da się zainstalować | uv (menedżer Pythona) |
| **[Plan C — Google Colab](docs/PLAN_C_COLAB.md)** | Ostateczność — gdy nic innego nie działa | Tylko przeglądarka |


## Szybki start (Plan A)

Jeśli masz już zainstalowany Docker Desktop ([instrukcja instalacji](docs/INSTALACJA_DOCKER_DESKTOP.md)):

```bash
git clone https://github.com/NXTRSS/MachineLearningCourse
cd MachineLearningCourse
docker compose up
```

Otwórz przeglądarkę: **http://localhost:8888** — gotowe!


## Wymagania sprzętowe

- **RAM**: minimum 6 GB (zalecane 8 GB)
- **Dysk**: ~5 GB wolnego miejsca (na Docker + obraz)
- **System**: Windows 10/11, macOS 12+, lub Linux (Ubuntu 20.04+)


## Weryfikacja środowiska

Po uruchomieniu środowiska (dowolnym planem) otwórz terminal w JupyterLab i uruchom:

```bash
python verify_env.py
```

Jeśli zobaczysz `Environment verification: OK` — wszystko jest gotowe!


## Dane do zajęć

Dane do zajęć zostaną pobrane automatycznie przy pierwszym uruchomieniu odpowiednich notebooków (przez skrypt `utils.py`). Jeśli automatyczne pobieranie nie zadziała, dane będą dostępne na dysku Google — link zostanie udostępniony na zajęciach.


## Problemy?

Jeśli żaden plan nie działa, proszę o kontakt mailowy przed zajęciami. Na zajęciach w ostateczności skorzystamy z Google Colab (Plan C).
