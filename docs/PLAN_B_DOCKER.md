# Plan B — Docker

Docker pozwala na uruchomienie gotowego środowiska w jednej komendzie, niezależnie od systemu operacyjnego.
Nie trzeba rozumieć jak Docker działa — wystarczy go zainstalować i uruchomić jedną komendę.


## Krok 1: Instalacja Docker Desktop

Jeśli Docker Desktop nie jest jeszcze zainstalowany, proszę przejść do osobnej instrukcji:

**➡️ [Instrukcja instalacji Docker Desktop](INSTALACJA_DOCKER_DESKTOP.md)**

Po instalacji proszę wrócić tutaj.


## Krok 2: Sprawdzenie czy Docker działa

Otwórz terminal (PowerShell na Windowsie, Terminal na Macu/Linuksie) i wpisz:

```bash
docker --version
```

Powinno się wyświetlić coś w stylu: `Docker version 27.x.x, build ...`

Jeśli pojawia się błąd — upewnij się, że Docker Desktop jest uruchomiony (ikonka w zasobniku systemowym / pasku menu).


## Krok 3: Pobranie plików z zajęć

```bash
git clone https://github.com/NXTRSS/MachineLearningCourse
cd MachineLearningCourse
```

Jeśli nie masz `git`, możesz też:
1. Wejść na https://github.com/NXTRSS/MachineLearningCourse
2. Kliknąć zielony przycisk **Code** → **Download ZIP**
3. Wypakować archiwum i otworzyć terminal w tym folderze


## Krok 4: Uruchomienie środowiska

W terminalu, będąc w folderze `MachineLearningCourse`, wpisz:

```bash
docker compose up
```

**Przy pierwszym uruchomieniu** Docker pobierze i zbuduje obraz — to może potrwać **5-15 minut** (w zależności od szybkości internetu). Przy kolejnych uruchomieniach będzie to kwestia sekund.

W terminalu pojawią się logi. Poczekaj, aż zobaczysz linijkę podobną do:

```
jupyter  |     http://127.0.0.1:8888/lab
```


## Krok 5: Otwarcie JupyterLab

Otwórz przeglądarkę i wejdź na adres:

**http://localhost:8888**

Zobaczysz JupyterLab z plikami z zajęć. Gotowe!


## Krok 6: Weryfikacja

W JupyterLab otwórz **Terminal** (File → New → Terminal) i wpisz:

```bash
python verify_env.py
```

Jeśli zobaczysz `Environment verification: OK` — środowisko jest poprawnie skonfigurowane.


## Rozpoczęcie pracy na kolejny dzień

Przy każdym kolejnym uruchomieniu wystarczy:

1. Upewnić się, że **Docker Desktop jest uruchomiony**
2. Otworzyć terminal w folderze `MachineLearningCourse`
3. Wpisać:
   ```bash
   docker compose up
   ```
4. Otworzyć **http://localhost:8888** w przeglądarce

Aby zatrzymać środowisko: naciśnij `Ctrl+C` w terminalu, lub wpisz w innym terminalu:

```bash
docker compose down
```


## Rozwiązywanie problemów

### "docker: command not found"
Docker Desktop nie jest zainstalowany lub nie jest uruchomiony. Sprawdź czy widzisz ikonkę Dockera w zasobniku systemowym.

### "port 8888 already in use"
Inny program (np. Jupyter) używa portu 8888. Zamknij go lub zmień port:
```bash
docker compose up -e JUPYTER_PORT=8889
```
Wtedy JupyterLab będzie na **http://localhost:8889**

### "docker compose" nie działa, ale "docker-compose" tak
Na starszych wersjach Dockera komenda nazywa się `docker-compose` (z myślnikiem):
```bash
docker-compose up
```

### Obraz buduje się bardzo długo
Pierwsze budowanie pobiera ~2 GB danych. Upewnij się, że masz stabilne połączenie z internetem. Przy kolejnych uruchomieniach będzie szybko.

### Nie działa? Przejdź do Planu C
**➡️ [Plan C — Google Colab](PLAN_C_COLAB.md)**
