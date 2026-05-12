# Testowanie serwera LM Studio z innego komputera

Instrukcja dla prowadzącego: jak sprawdzić czy Twój LM Studio jest osiągalny z innego kompa w sieci LAN **przed zajęciami**.

---

## 1. Przygotowanie serwera (Twój Mac)

1. Uruchom **LM Studio**
2. Załaduj model (np. `qwen3:8b`)
3. Przejdź do zakładki **Local Server** (ikona `<->` po lewej)
4. Kliknij **Start Server**
5. Upewnij się, że nasłuchuje na `0.0.0.0:1234` (nie `127.0.0.1` !)
   - W ustawieniach serwera → "Serve on Local Network" musi być **włączone**
6. Sprawdź swój adres IP:
   ```bash
   ipconfig getifaddr en0
   ```
   Zapamiętaj wynik, np. `192.168.1.42`

---

## 2. Szybki test z tego samego kompa

```bash
curl http://localhost:1234/v1/models
```

Oczekiwany wynik — JSON z nazwą modelu:
```json
{"object":"list","data":[{"id":"qwen3-8b","object":"model",...}]}
```

Jeśli tu nie działa — problem z LM Studio, nie z siecią.

---

## 3. Test z drugiego komputera (w tym samym WiFi)

### macOS / Linux:
```bash
# Zamień IP na swoje z kroku 1
curl http://192.168.1.42:1234/v1/models
```

### Windows (PowerShell):
```powershell
Invoke-RestMethod -Uri "http://192.168.1.42:1234/v1/models"
```

### Test pełnego zapytania (chat completion):
```bash
curl http://192.168.1.42:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any",
    "messages": [{"role": "user", "content": "Powiedz cześć"}],
    "max_tokens": 50
  }'
```

Oczekiwany wynik — JSON z odpowiedzią modelu.

---

## 4. Jeśli nie działa — checklist

### a) Firewall macOS
```bash
# Sprawdź status firewalla
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
```
Jeśli firewall jest włączony:
- System Preferences → Network → Firewall → Options
- Dodaj LM Studio do listy aplikacji z "Allow incoming connections"
- **Albo** tymczasowo wyłącz firewall na czas zajęć:
  ```bash
  sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off
  ```
  (włącz potem z powrotem: `--setglobalstate on`)

### b) Zła sieć WiFi
Oba komputery muszą być w **tej samej** sieci. Sprawdź:
```bash
# Na Twoim Macu:
ipconfig getifaddr en0    # np. 192.168.1.42

# Na drugim kompie (macOS/Linux):
ipconfig getifaddr en0    # np. 192.168.1.XX  ← te same 3 pierwsze oktety
```
Jeśli pierwszy ma `192.168.1.x` a drugi `192.168.0.x` — są w różnych sieciach.

### c) LM Studio nasłuchuje tylko na localhost
W LM Studio → Server Settings:
- **Host:** musi być `0.0.0.0` (nie `127.0.0.1`)
- **Port:** `1234`
- "Serve on Local Network" → **ON**

### d) Port zablokowany przez router / AP
Rzadkie w domowych sieciach, ale hotspot telefonu może izolować klientów ("AP Isolation").
Jeśli testujesz przez hotspot — wyłącz "Client Isolation" w ustawieniach hotspota.

---

## 5. Test obciążeniowy (opcjonalny)

Symulacja kilku studentów naraz:
```bash
# Odpal 5 równoczesnych zapytań
for i in {1..5}; do
  curl -s http://192.168.1.42:1234/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"any","messages":[{"role":"user","content":"Ile jest 2+2?"}],"max_tokens":20}' &
done
wait
echo "Wszystkie zapytania zakończone"
```

---

## 6. Test z Dockera

Jeśli student pracuje w Dockerze, połączenie z hostem wymaga specjalnego adresu:
```bash
# Wewnątrz kontenera Docker:
curl http://host.docker.internal:1234/v1/models
```
`host.docker.internal` to adres hosta z perspektywy kontenera (Docker Desktop na Mac/Windows).

Na Linuxie trzeba dodać `--add-host=host.docker.internal:host-gateway` do `docker run`.

---

## 7. Tunnel (dostęp spoza LAN / Colab)

Jeśli chcesz wystawić serwer na zewnątrz (np. dla studentów na Google Colab):

### ngrok (najprostszy):
```bash
# Instalacja (jednorazowo)
brew install ngrok

# Uruchomienie tunnelu
ngrok http 1234
```
Dostaniesz publiczny URL (np. `https://abc123.ngrok-free.app`). Studenci używają go zamiast `http://192.168.x.x:1234`.

### Cloudflare Tunnel (darmowy, stabilniejszy):
```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:1234
```

### Uwagi dot. tunneli:
- **Latencja** — dodaje 20-100ms, ale dla LLM (sekundy generacji) to nieistotne
- **Bezpieczeństwo** — tunnel wystawia Twój serwer publicznie! Każdy z linkiem może wysyłać zapytania
- **ngrok free tier** — limit połączeń, URL zmienia się po restarcie
- **Cloudflare** — darmowy, URL też tymczasowy ale bardziej stabilny
- Na zajęciach z Colab podaj studentom URL tunnelu zamiast IP LAN
