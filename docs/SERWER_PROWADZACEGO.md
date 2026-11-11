# Podłączenie do serwera prowadzącego

Na zajęciach z Function Calling i RAG możesz użyć modelu językowego hostowanego przez prowadzącego zamiast instalować go lokalnie.

## Wymagania

- Laptop w **tej samej sieci WiFi** co prowadzący
- Środowisko uruchomione (Plan A lub B — patrz główne README)
- Adres serwera i hasło — poda prowadzący na zajęciach (format adresu: `http://ADRES:PORT`, np. `http://192.168.X.X:4242`)

Przy łączeniu zostaniesz zapytany o **imię** (prowadzący widzi je na swoim ekranie, żeby wiedzieć, kto czeka na odpowiedź) i o **hasło** od prowadzącego.


## Szybki test połączenia

Otwórz notebook **`test_polaczenia.ipynb`**, wpisz adres podany przez prowadzącego i uruchom komórkę (zapyta o imię i hasło). Jeśli zobaczysz `✓ Wszystko działa!` — gotowe.


## Najczęstsze problemy

### "Nie można połączyć się z serwerem zdalnym"

**Przyczyna 1 — zła sieć WiFi**

Sprawdź czy Twój laptop jest w tej samej sieci co prowadzący. Twój adres IP:
- Windows: `ipconfig` → szukaj `IPv4 Address`
- macOS: `ipconfig getifaddr en0`
- Linux: `hostname -I`

Adres powinien zaczynać się od tych samych trzech liczb co adres prowadzącego (np. oba `192.168.0.x`).

**Przyczyna 2 — wpisujesz `https` zamiast `http`**

Adres serwera **zawsze** zaczyna się od `http://`, nigdy `https://`. To częsty błąd.

**Przyczyna 3 — literówka w adresie**

Przepisz adres **dokładnie** tak, jak podał prowadzący — razem z portem po dwukropku (np. `:4242`).

### „Nieprawidłowe hasło” / błąd 401 albo 429

- **„Ten serwer wymaga hasła od prowadzącego”** — uruchom komórkę z `setup_auth_client` (albo `test_polaczenia.ipynb`) i wpisz hasło, gdy notebook o nie zapyta.
- **„Nieprawidłowe hasło”** — wpisz je dokładnie tak, jak podał prowadzący (wielkość liter ma znaczenie). Notebook pozwala spróbować 3 razy, potem uruchom komórkę ponownie.
- **„Zbyt wiele błędnych haseł”** (błąd 429) — po 5 błędnych próbach serwer blokuje Twój komputer na ok. minutę. Odczekaj i spróbuj z poprawnym hasłem.

### „Brak imienia studenta” / błąd 400

Serwer prowadzącego wymaga imienia (prowadzący widzi, kto czeka w kolejce). Uruchom ponownie komórkę z `setup_auth_client` — zapyta o imię. Polskie znaki w imieniu (np. „Łukasz”) są w porządku. Jeśli błąd się powtarza: `git pull` i **Kernel → Restart Kernel** (stary `utils.py` w pamięci kernela nie wysyła imienia).

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
Invoke-RestMethod -Uri "http://192.168.X.X:4242/v1/models"
```

---

### Długi czas oczekiwania na odpowiedź

Jeden serwer obsługuje całą grupę — przy wielu równoczesnych zapytaniach kolejka jest normalna. **Nie klikaj wielokrotnie** — każde kliknięcie dodaje kolejne zapytanie do kolejki i wydłuża czas oczekiwania dla wszystkich.


## Weryfikacja połączenia z wiersza poleceń

macOS/Linux:
```bash
curl http://<adres-prowadzącego>:<port>/v1/models
```

Windows (PowerShell):
```powershell
Invoke-RestMethod -Uri "http://<adres-prowadzącego>:<port>/v1/models"
```

Jeśli serwer działa, dostaniesz odpowiedź JSON z nazwą modelu.
