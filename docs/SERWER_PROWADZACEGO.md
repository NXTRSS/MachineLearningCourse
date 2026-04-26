# Podłączenie do serwera prowadzącego

Na zajęciach z Function Calling i RAG możesz użyć modelu językowego hostowanego przez prowadzącego zamiast instalować go lokalnie.

## Wymagania

- Laptop w **tej samej sieci WiFi** co prowadzący
- Środowisko uruchomione (Plan A lub B — patrz główne README)
- Adres IP serwera podany przez prowadzącego (format: `http://192.168.X.X:1234`)


## Szybki test połączenia

Otwórz notebook **`test_polaczenia.ipynb`**, wpisz adres podany przez prowadzącego i uruchom obie komórki. Jeśli zobaczysz `✓ Połączenie OK!` — gotowe.


## Najczęstsze problemy

### "Nie można połączyć się z serwerem zdalnym"

**Przyczyna 1 — zła sieć WiFi**

Sprawdź czy Twój laptop jest w tej samej sieci co prowadzący. Twój adres IP:
- Windows: `ipconfig` → szukaj `IPv4 Address`
- macOS/Linux: `ipconfig getifaddr en0`

Adres powinien zaczynać się od tych samych trzech liczb co adres prowadzącego (np. oba `192.168.0.x`).

**Przyczyna 2 — wpisujesz `https` zamiast `http`**

Adres serwera **zawsze** zaczyna się od `http://`, nigdy `https://`. To częsty błąd.

**Przyczyna 3 — zły port**

Serwer prowadzącego działa na porcie `1234` (LM Studio), nie `11434` (Ollama). Upewnij się że adres kończy się `:1234`.

---

### "Brak dostępnego LLM-a" w notebooku

Notebook nie znalazł żadnego modelu. Możliwe przyczyny:

1. **Adres IP jest stary** — prowadzący mógł się przełączyć na inną sieć. Zapytaj o aktualny adres.
2. **Model nie jest załadowany** — prowadzący musi załadować model w LM Studio. Poczekaj chwilę i uruchom komórkę ponownie.
3. **Nie zaktualizowałeś repo** — uruchom `git pull` i zrestartuj kernel Jupytera (Kernel → Restart Kernel and Run All Cells).

---

### Notebook wykrywa zły model (np. stary)

Zrestartuj kernel: **Kernel → Restart Kernel and Run All Cells**. Bez restartu Jupyter używa wartości z poprzedniego uruchomienia.

---

### Windows: `curl` nie działa

Na Windowsie `curl` w PowerShell działa inaczej niż na macOS. Zamiast:
```
curl http://...
```
użyj:
```powershell
Invoke-RestMethod -Uri "http://192.168.X.X:1234/v1/models"
```

---

### Długi czas oczekiwania na odpowiedź

Jeden serwer obsługuje całą grupę — przy wielu równoczesnych zapytaniach kolejka jest normalna. **Nie klikaj wielokrotnie** — każde kliknięcie dodaje kolejne zapytanie do kolejki i wydłuża czas oczekiwania dla wszystkich.


## Weryfikacja połączenia z wiersza poleceń

macOS/Linux:
```bash
curl http://<adres-prowadzącego>:1234/v1/models
```

Windows (PowerShell):
```powershell
Invoke-RestMethod -Uri "http://<adres-prowadzącego>:1234/v1/models"
```

Jeśli serwer działa, dostaniesz odpowiedź JSON z nazwą modelu.
