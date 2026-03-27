# Instalacja Docker Desktop

Docker Desktop to aplikacja, która pozwala uruchamiać kontenery (gotowe, zamknięte środowiska) na komputerze. Instalacja jest jednorazowa.


## Windows

### Wymagania
- Windows 10 (wersja 1903+) lub Windows 11
- Włączony WSL2 (Windows Subsystem for Linux)

### Krok 1: Włączenie WSL2

Otwórz **PowerShell jako administrator** (kliknij prawym przyciskiem na Start → "Windows PowerShell (Administrator)" lub "Terminal (Administrator)") i wpisz:

```powershell
wsl --install
```

Zrestartuj komputer po zakończeniu instalacji.

Po restarcie, otwórz ponownie PowerShell i sprawdź:

```powershell
wsl --version
```

Powinno wyświetlić informacje o zainstalowanej wersji WSL.

**Jeśli `wsl --install` nie zadziała**, sprawdź szczegółową instrukcję Microsoftu:
https://learn.microsoft.com/pl-pl/windows/wsl/install

### Krok 2: Pobranie Docker Desktop

1. Wejdź na: https://www.docker.com/products/docker-desktop/
2. Kliknij **"Download for Windows"**
3. Uruchom pobrany plik `Docker Desktop Installer.exe`
4. Podczas instalacji zaznacz opcję **"Use WSL 2 instead of Hyper-V"** (powinna być domyślnie zaznaczona)
5. Kliknij **OK** / **Install** / **Close and restart**

### Krok 3: Pierwsze uruchomienie

1. Po restarcie komputera uruchom **Docker Desktop** (z menu Start)
2. Zaakceptuj warunki licencji
3. Poczekaj aż Docker się uruchomi (ikonka wieloryba w zasobniku systemowym przestanie się animować)
4. Otwórz PowerShell i sprawdź: `docker --version`

### Wizualny poradnik instalacji na Windows

Jeśli powyższe kroki sprawiają trudności, polecam ten poradnik wideo (po angielsku, ale kroki są identyczne):

- Oficjalna dokumentacja Docker z grafikami: https://docs.docker.com/desktop/setup/install/windows-install/


---


## macOS

### Wymagania
- macOS 12 (Monterey) lub nowszy
- Procesor Intel lub Apple Silicon (M1/M2/M3/M4)

### Instalacja

1. Wejdź na: https://www.docker.com/products/docker-desktop/
2. Kliknij **"Download for Mac"**
   - Strona powinna automatycznie wykryć typ procesora (Intel / Apple Silicon)
   - Jeśli nie wiesz jaki masz procesor: kliknij logo Apple →  "About This Mac" → przy "Chip" będzie albo "Apple M..." albo "Intel"
3. Otwórz pobrany plik `.dmg`
4. Przeciągnij ikonkę Docker do folderu **Applications**
5. Uruchom Docker z folderu Applications
6. Zaakceptuj warunki licencji
7. Poczekaj aż Docker się uruchomi (ikonka wieloryba na pasku menu u góry)
8. Otwórz Terminal i sprawdź: `docker --version`

### Wizualny poradnik instalacji na macOS

- Oficjalna dokumentacja Docker z grafikami: https://docs.docker.com/desktop/setup/install/mac-install/


---


## Linux (Ubuntu / Debian)

Na Linuksie można zainstalować Docker Engine bezpośrednio (bez Docker Desktop), co jest lżejsze.

### Instalacja Docker Engine

Otwórz terminal i wykonaj po kolei:

```bash
# Aktualizacja pakietów
sudo apt-get update

# Instalacja wymaganych zależności
sudo apt-get install -y ca-certificates curl gnupg

# Dodanie klucza GPG Dockera
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Dodanie repozytorium Dockera
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Instalacja Dockera
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Dodanie użytkownika do grupy docker (żeby nie trzeba było sudo)
sudo usermod -aG docker $USER
```

**Po wykonaniu ostatniej komendy wyloguj się i zaloguj ponownie** (lub zrestartuj komputer).

Sprawdź instalację:

```bash
docker --version
docker compose version
```

### Oficjalna dokumentacja

- https://docs.docker.com/engine/install/ubuntu/


---


## Weryfikacja instalacji (wszystkie systemy)

Po zainstalowaniu Dockera otwórz terminal i wpisz:

```bash
docker run hello-world
```

Jeśli zobaczysz tekst zaczynający się od `Hello from Docker!` — instalacja się powiodła.
