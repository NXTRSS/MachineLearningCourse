# Test Playbook — Docker & Colab LLM Access

Scenariusze testowe dla zmian na branchu `docker-colab-llm-access`.
Potrzebujesz: swój Mac + jeden inny komputer (lub telefon w tej samej sieci WiFi).

---

## Wymagania

- LM Studio z załadowanym modelem (np. `gemma4:e4b`) na Twoim Macu
- Docker Desktop zainstalowany i uruchomiony
- `cloudflared` zainstalowany (`brew install cloudflared`)
- Drugi komputer lub telefon w tej samej sieci WiFi (do testów LAN)
- Konto Google (do testu Colab)

---

## Test 1: Proxy — domyślnie localhost-only

**Cel:** Proxy bez `--lan` nie jest dostępne z sieci.

```bash
# Terminal 1: uruchom proxy
python llm_proxy.py -s test123

# Oczekiwany output:
#   🔀 Proxy nasłuchuje na 127.0.0.1:4242
#   🔒 Tylko localhost (dodaj --lan aby otworzyć dla studentów w sieci)
```

**Weryfikacja z Maca:**
```bash
# Powinno działać (localhost):
curl -s -H "Authorization: Bearer test123" http://localhost:4242/v1/models
# Oczekiwany: JSON z modelami

# Sprawdź swoje IP:
ipconfig getifaddr en0   # np. 192.168.1.100

# Powinno NIE działać (LAN IP):
curl -s http://192.168.1.100:4242/v1/models
# Oczekiwany: Connection refused
```

**Weryfikacja z drugiego komputera / telefonu:**
```
# W przeglądarce telefonu wejdź na:
http://192.168.1.100:4242/v1/models
# Oczekiwany: nie ładuje się (connection refused)
```

**PASS jeśli:** localhost działa, LAN IP nie.

---

## Test 2: Proxy — tryb LAN

**Cel:** Z `--lan` proxy jest dostępne w sieci lokalnej.

```bash
# Terminal 1: uruchom proxy z --lan
python llm_proxy.py --lan -s test123
# Oczekiwany:
#   🔀 Proxy nasłuchuje na 0.0.0.0:4242
#   📋 Adres dla studentów (LAN):
#      LECTURER_SERVER = "http://192.168.x.x:4242"
```

**Weryfikacja z drugiego komputera:**
```bash
# /v1/models — bez hasła (health check):
curl -s http://192.168.x.x:4242/v1/models
# Oczekiwany: JSON z modelami

# /v1/chat/completions — bez hasła:
curl -s -X POST http://192.168.x.x:4242/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"hi"}]}'
# Oczekiwany: 401 Unauthorized

# /v1/chat/completions — z hasłem:
curl -s -X POST http://192.168.x.x:4242/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test123" \
  -d '{"model":"gemma-4-e4b-it-mlx","messages":[{"role":"user","content":"Powiedz cześć"}]}'
# Oczekiwany: odpowiedź modelu
```

**Weryfikacja dashboardu:**
```
# Na swoim Macu otwórz: http://localhost:5050
# Powinien działać

# Z drugiego komputera: http://192.168.x.x:5050
# Powinien NIE działać (dashboard zawsze localhost)
```

**PASS jeśli:** LAN dostęp do proxy działa z hasłem, dashboard tylko z localhost.

---

## Test 3: Proxy — rate limiting

**Cel:** 5 złych haseł z tego samego IP = blokada na 60s.

```bash
# Uruchom proxy z --lan:
python llm_proxy.py --lan -s prawidlowe-haslo

# Z drugiego terminala — 5 razy złe hasło:
for i in 1 2 3 4 5; do
  curl -s -X POST http://localhost:4242/v1/chat/completions \
    -H "Authorization: Bearer zle-haslo" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[]}'
  echo " --- próba $i"
done

# 6. próba — nawet z prawidłowym hasłem:
curl -s -X POST http://localhost:4242/v1/chat/completions \
  -H "Authorization: Bearer prawidlowe-haslo" \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[]}'
# Oczekiwany: 429 "Zbyt wiele nieudanych prób"
```

**PASS jeśli:** po 5 złych próbach — blokada, nawet prawidłowe hasło nie przechodzi.

---

## Test 4: Tunnel (Cloudflare)

**Cel:** `--tunnel` tworzy publiczny URL, hasło wymagane.

```bash
# Bez hasła — powinien wymusić:
python llm_proxy.py --tunnel
# Oczekiwany: "⚠️ --tunnel wymaga --student-key"

# Z hasłem:
python llm_proxy.py --tunnel -s alk-test-2026
# Oczekiwany:
#   🌐 Uruchamiam Cloudflare Tunnel...
#   ✅ Tunel aktywny!
#   🌐 URL dla Colab / zdalnych studentów:
#      LECTURER_SERVER = "https://xxx-yyy.trycloudflare.com"
```

**Weryfikacja z dowolnego komputera/telefonu:**
```bash
# Health check (bez hasła):
curl -s https://xxx-yyy.trycloudflare.com/v1/models
# Oczekiwany: JSON z modelami

# Chat bez hasła:
curl -s -X POST https://xxx-yyy.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[]}'
# Oczekiwany: 401

# Chat z hasłem:
curl -s -X POST https://xxx-yyy.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer alk-test-2026" \
  -d '{"model":"gemma-4-e4b-it-mlx","messages":[{"role":"user","content":"Cześć"}]}'
# Oczekiwany: odpowiedź modelu
```

**PASS jeśli:** tunel startuje, URL działa z internetu, auth wymagany.

> **Uwaga:** Z samym `--tunnel` (bez `--lan`) proxy słucha na localhost.
> Tunel działa bo cloudflared łączy się do localhost. Ale nikt z LAN nie sięgnie bezpośrednio.

---

## Test 5: Docker — connect_llm wykrywa host LLM

**Cel:** Notebook w Dockerze automatycznie znajduje LM Studio na hoście.

```bash
# Upewnij się, że LM Studio działa na Macu z załadowanym modelem

# Zbuduj i uruchom Docker:
docker compose up -d --build

# Test connect_llm z kontenera:
docker compose exec jupyter python -c "
from utils import _is_docker, connect_llm
print('Docker:', _is_docker())
client, instr, model = connect_llm()
if client:
    print(f'SUCCESS: model={model}')
    # Szybki test — czy model odpowiada:
    r = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': 'Powiedz tylko: OK'}],
        max_tokens=10,
    )
    print(f'Response: {r.choices[0].message.content}')
else:
    print('FAIL: no LLM found')
"

# Zatrzymaj:
docker compose down
```

**PASS jeśli:** `Docker: True`, `SUCCESS: model=...`, odpowiedź od modelu.

---

## Test 6: Docker — connect_llm z proxy prowadzącego

**Cel:** Docker student łączy się z proxy zamiast bezpośrednio z LM Studio.

```bash
# Terminal 1: uruchom proxy z --lan
python llm_proxy.py --lan -s alk-2026

# Terminal 2: Docker
docker compose up -d
docker compose exec jupyter python -c "
from utils import connect_llm
client, instr, model = connect_llm(
    lecturer_server='http://host.docker.internal:4242',
    api_key='alk-2026',
)
if client:
    print(f'SUCCESS via proxy: {model}')
else:
    print('FAIL')
"
docker compose down
```

**PASS jeśli:** `SUCCESS via proxy: ...`

---

## Test 7: Docker — setup_auth_client z proxy

**Cel:** Docker student dostaje prompt o hasło (jak na prawdziwych zajęciach).

```bash
# Terminal 1: proxy z --lan i hasłem
python llm_proxy.py --lan -s alk-2026

# Terminal 2: Docker — interaktywny test
docker compose up -d
docker compose exec -it jupyter python -c "
from utils import connect_llm, setup_auth_client
client, instr, model = connect_llm(
    lecturer_server='http://host.docker.internal:4242',
)
print(f'Model: {model}')
# Teraz setup_auth_client powinien zapytać o imię i hasło:
client, instr = setup_auth_client(client, instr, model)
if client:
    print('SUCCESS: auth OK')
else:
    print('FAIL: auth failed')
"
docker compose down
```

**Interakcja:**
- `👤 Twoje imię:` → wpisz np. `TestStudent`
- `🔑 Hasło:` → wpisz `alk-2026`

**Weryfikacja na dashboardzie:** otwórz `http://localhost:5050` — powinien widać `TestStudent`.

**PASS jeśli:** auth działa, student widoczny na dashboardzie.

---

## Test 8: Google Colab — połączenie przez tunel

**Cel:** Notebook na Colabie łączy się z proxy przez Cloudflare Tunnel.

```bash
# Na swoim Macu:
python llm_proxy.py --tunnel -s alk-colab-test
# Zapisz URL tunelu, np. https://abc-xyz.trycloudflare.com
```

**W Google Colab (colab.research.google.com):**

Utwórz nowy notebook i wklej w komórki:

```python
# Komórka 1:
!pip install openai

# Komórka 2:
from openai import OpenAI

TUNNEL_URL = "https://abc-xyz.trycloudflare.com"  # ← wpisz swój URL
HASLO = "alk-colab-test"

client = OpenAI(base_url=f"{TUNNEL_URL}/v1", api_key=HASLO)

# Test 1: lista modeli
models = [m.id for m in client.models.list().data]
print(f"Modele: {models}")

# Test 2: chat
r = client.chat.completions.create(
    model=models[0],
    messages=[{"role": "user", "content": "Powiedz tylko: Cześć z Colaba!"}],
    max_tokens=20,
)
print(f"Odpowiedź: {r.choices[0].message.content}")
```

**Weryfikacja na dashboardzie:** `http://localhost:5050` — powinno pojawić się zapytanie.

**PASS jeśli:** Colab dostaje odpowiedź od modelu, widać zapytanie na dashboardzie.

---

## Test 9: Colab — pełny flow z notebookiem FC

**Cel:** Function_Calling.ipynb działa z Colaba (end-to-end).

1. Wgraj `Function_Calling.ipynb` i `utils.py` do Colab
2. W komórce z `LECTURER_SERVER` zmień na URL tunelu
3. Uruchom komórki po kolei — `connect_llm` powinien wykryć serwer
4. `setup_auth_client` powinien poprosić o imię i hasło
5. Przetestuj przynajmniej jedno function calling (np. pogoda, kalkulator)

**PASS jeśli:** FC działa end-to-end z Colaba.

---

## Test 10: Ctrl+C — graceful shutdown

**Cel:** Proxy, dashboard i tunel zamykają się czysto.

```bash
python llm_proxy.py --tunnel --lan -s test123
# Poczekaj aż tunel będzie aktywny
# Naciśnij Ctrl+C

# Oczekiwany output:
#   Proxy zatrzymane.
#   Łącznie: X zapytań, ...
#   Tunel zamknięty.
```

**Weryfikacja:** po Ctrl+C sprawdź, że żadne procesy nie zostały:
```bash
# Nie powinno nic znaleźć:
ps aux | grep cloudflared | grep -v grep
lsof -i :4242
lsof -i :5050
```

**PASS jeśli:** czyste zamknięcie, brak zombie procesów.

---

## Szybka ściągawka — minimalny zestaw testów

Jeśli masz mało czasu, zrób przynajmniej te 4:

| # | Test | Czas | Co sprawdza |
|---|------|------|------------|
| 1 | Proxy localhost-only | 2 min | Domyślne bezpieczeństwo |
| 5 | Docker + host LLM | 3 min | Główny flow Docker studentów |
| 4 | Tunnel start | 2 min | Cloudflare działa |
| 8 | Colab + tunel | 5 min | Główny flow Colab studentów |
