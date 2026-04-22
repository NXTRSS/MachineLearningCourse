# TODO: Serwer prowadzącego przez ngrok (dla studentów na Colabie)

## Problem który rozwiązujemy

Studenci na Google Colab nie mają dostępu do lokalnej sieci WiFi na sali,
więc nie mogą połączyć się z serwerem Ollamy prowadzącego przez lokalny IP
(np. `192.168.1.100:11434`). Ngrok rozwiązuje to przez publiczny tunel.

## Jak to ma działać docelowo

```
Prowadzący (MacBook z Ollamą + ngrok)
    → ngrok wystawia publiczny URL, np. https://abc123.ngrok-free.app
    → prowadzący wpisuje ten URL na tablicy / wkleja na czacie
    → studenci wpisują URL do INSTRUCTOR_SERVER w notebooku
    → działa zarówno z lokalnego laptopa jak i z Colaba
```

## Co trzeba zrobić (krok po kroku)

### 1. Instalacja ngrok (jednorazowo)

```bash
# macOS
brew install ngrok

# lub bez brew - pobierz ze strony:
# https://ngrok.com/download
```

Założyć darmowe konto na https://ngrok.com i podpiąć token:
```bash
ngrok config add-authtoken TWÓJ_TOKEN
```

### 2. Przed zajęciami — uruchomienie tunelu

Upewnij się że Ollama działa:
```bash
ollama serve   # jeśli nie chodzi jako daemon
ollama list    # sprawdź jakie modele są dostępne
```

Uruchom ngrok:
```bash
ngrok http 11434
```

Ngrok wypisze coś w stylu:
```
Forwarding   https://abc123.ngrok-free.app -> http://localhost:11434
```

Ten URL (`https://abc123.ngrok-free.app`) podajesz studentom.

### 3. Zmiany w notebookach (do zrobienia)

Obecnie `INSTRUCTOR_SERVER` jest hardcoded jako lokalny IP:
```python
INSTRUCTOR_SERVER = "http://192.168.1.100:11434"  # <-- stary sposób
```

Docelowo warto zmienić na coś takiego (w `Function_Calling_warsztat.ipynb`
i `RAG_warsztat.ipynb`):

```python
# Wpisz adres podany przez prowadzącego (lokalny IP lub ngrok URL):
INSTRUCTOR_SERVER = "http://192.168.1.100:11434"   # w sieci sali
# INSTRUCTOR_SERVER = "https://abc123.ngrok-free.app"  # przez ngrok (Colab)
```

Ewentualnie dodać auto-detekcję: jeśli URL zaczyna się od `https://` to
nie dodawać `/api/tags` tylko używać endpointu ngrok bezpośrednio.

**Uwaga:** ngrok wymaga nagłówka `ngrok-skip-browser-warning: true` w
requestach HTTP żeby ominąć stronę powitalną. Trzeba dodać do `detect_ollama`:

```python
def detect_ollama(base_url="http://localhost:11434"):
    headers = {}
    if "ngrok" in base_url:
        headers["ngrok-skip-browser-warning"] = "true"
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5, headers=headers)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except:
        pass
    return []
```

Ten sam fix potrzebny w `detect_lmstudio`.

### 4. Zmiany w setup_local_llm.ipynb (do zrobienia)

Dodać sekcję "Plan C — serwer prowadzącego" z instrukcją:
- jak wpisać URL podany przez prowadzącego
- wzmianka że URL zaczyna się od `https://` gdy prowadzący używa ngrok

### 5. Zmiany w README (do zrobienia)

W sekcji "Ollama (warsztaty LLM)" dodać:
- wzmiankę o serwerze prowadzącego
- że na Colabie trzeba użyć URL ngrok zamiast lokalnego IP
- że URL zostanie podany na zajęciach

### 6. Opcjonalnie — stały URL (płatny plan ngrok ~$10/mies.)

Darmowy ngrok daje losowy URL przy każdym uruchomieniu. Płatny plan
pozwala ustawić stały subdomain, np. `https://alk-ml.ngrok.io`.
Wtedy URL można wpisać na stałe do notebooka i nie trzeba go podawać
na każdych zajęciach.

## Pliki do zmodyfikowania

- [ ] `Function_Calling_warsztat.ipynb` — fix `detect_ollama` + komentarz przy `INSTRUCTOR_SERVER`
- [ ] `RAG_warsztat.ipynb` — to samo
- [ ] `setup_local_llm.ipynb` — dodać Plan C z instrukcją wpisania URL
- [ ] `README.md` — wzmianka o ngrok dla Colaba
- [ ] `docs/PLAN_C_COLAB.md` — dodać sekcję o serwerze prowadzącego przez ngrok

## Kontekst sesji (dla Claude)

Ta sesja (kwiecień 2026) dotyczyła przebudowy setupu kursu ML z Minicondy
na trójwarstwowe podejście: Docker / uv / Colab. Wszystkie zmiany są na
gałęzi `docker-uv-setup` repozytorium `NXTRSS/MachineLearningCourse`.

Notebooki LLM (`Function_Calling_warsztat.ipynb`, `RAG_warsztat.ipynb`)
używają OpenAI-compatible API przez Ollamę lub LM Studio.
Auto-detekcja sprawdza kolejno: Ollama (port 11434) → LM Studio (port 1234)
→ serwer prowadzącego → błąd.
