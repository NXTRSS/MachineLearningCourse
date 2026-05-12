# Docker & Colab LLM Access — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Docker (Plan B) students to use either their own local LLM or the lecturer's proxy, and enable Colab (Plan C) students to connect to the lecturer's proxy via internet tunnel.

**Architecture:** Three changes:
1. Docker-compose gets `extra_hosts` so containers can reach host services via `host.docker.internal`
2. `connect_llm()` in `utils.py` learns to auto-detect Docker environment and adjust URLs
3. `llm_proxy.py` gains a `--tunnel` flag that launches `cloudflared` as subprocess, printing the public URL for Colab students

**Tech Stack:** Docker Compose, Cloudflare Quick Tunnel (`cloudflared`), Python stdlib (subprocess), OpenAI SDK

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `docker-compose.yml` | Modify | Add `extra_hosts` for host.docker.internal |
| `utils.py` | Modify | `connect_llm()` auto-detects Docker, tries `host.docker.internal` |
| `llm_proxy.py` | Modify | Add `--tunnel` flag, `cloudflared` subprocess management |
| `docs/PLAN_B_DOCKER.md` | Modify | Document Docker LLM access (host LLM + lecturer proxy) |
| `docs/PLAN_C_COLAB.md` | Modify | Document Colab connection via tunnel URL |
| `docs/SERWER_PROWADZACEGO.md` | Modify | Add tunnel section for remote students |
| `docs/LOKALNE_LLM.md` | Modify | Add Docker-specific note |

---

### Task 1: docker-compose.yml — add `extra_hosts`

**Files:**
- Modify: `docker-compose.yml`

This single line enables the Jupyter container to reach any service running on the host machine (LM Studio on port 1234, Ollama on 11434, or the lecturer's proxy on 4242) via the hostname `host.docker.internal`. On Docker Desktop (Mac/Windows) this resolves automatically, but the explicit `extra_hosts` line makes it work on native Linux Docker too.

- [ ] **Step 1: Add extra_hosts to docker-compose.yml**

Current `docker-compose.yml`:
```yaml
services:
  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - .:/app
      - pip-cache:/root/.cache/pip
    environment:
      - JUPYTER_ENABLE_LAB=yes

volumes:
  pip-cache:
```

Add `extra_hosts` under the `jupyter` service. The modified file should be:
```yaml
services:
  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - .:/app
      - pip-cache:/root/.cache/pip
    environment:
      - JUPYTER_ENABLE_LAB=yes
    extra_hosts:
      - "host.docker.internal:host-gateway"

volumes:
  pip-cache:
```

- [ ] **Step 2: Verify the change works**

Build and run the container, then verify DNS resolution inside:
```bash
docker compose up -d
docker compose exec jupyter python -c "import socket; print(socket.gethostbyname('host.docker.internal'))"
docker compose down
```

Expected: prints an IP address (e.g., `172.17.0.1` on Linux, `192.168.65.254` on Mac)

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "$(cat <<'EOF'
docker: add extra_hosts so container can reach host services

Enables Docker students to access LM Studio, Ollama, or the
lecturer's proxy running on their host machine via
host.docker.internal. Works on Docker Desktop (Mac/Windows)
and native Linux Docker.
EOF
)"
```

---

### Task 2: `utils.py` — auto-detect Docker and adjust `connect_llm`

**Files:**
- Modify: `utils.py:650-788` (the `connect_llm` function)

The key insight: inside Docker, `localhost` means the container itself (nothing there), but `host.docker.internal` reaches the host. We need `connect_llm` to automatically try `host.docker.internal` ports when running inside Docker, so students don't have to change any code.

Detection strategy: check if `/.dockerenv` exists (Docker creates this file in every container).

- [ ] **Step 1: Add Docker detection helper**

Add this function just before `connect_llm` (around line 649, after `pick_best_model`):

```python
def _is_docker():
    """Detect if running inside a Docker container."""
    from pathlib import Path
    return Path("/.dockerenv").exists()
```

- [ ] **Step 2: Modify `_try_lmstudio` to try `host.docker.internal` in Docker**

Inside `connect_llm`, modify the `_try_lmstudio` inner function. Currently it only tries `localhost` ports. When in Docker, it should also try the same ports on `host.docker.internal`.

Current code (lines ~711-722):
```python
    def _try_lmstudio():
        for port in lms_ports:
            url = f"http://localhost:{port}"
            print(f"Szukam LM Studio (port {port})...")
            models = detect_lmstudio(url, api_key=api_key)
            if not models and port == 1234:
                models = _try_launch_lms() and detect_lmstudio(url, api_key=api_key)
            if models:
                picked = _pick(models)
                print(f"✓ LM Studio (port {port})! Model: {picked}")
                return _make_clients(url, "lm-studio", picked)
        return None
```

Replace with:
```python
    def _try_lmstudio():
        in_docker = _is_docker()
        hosts = ["localhost"]
        if in_docker:
            hosts.append("host.docker.internal")
        for host in hosts:
            for port in lms_ports:
                url = f"http://{host}:{port}"
                label = f"port {port}" if host == "localhost" else f"{host}:{port}"
                print(f"Szukam LM Studio ({label})...")
                models = detect_lmstudio(url, api_key=api_key)
                if not models and host == "localhost" and port == 1234:
                    models = _try_launch_lms() and detect_lmstudio(url, api_key=api_key)
                if models:
                    picked = _pick(models)
                    print(f"✓ LM Studio ({label})! Model: {picked}")
                    return _make_clients(url, "lm-studio", picked)
        return None
```

- [ ] **Step 3: Modify `_try_ollama` to try `host.docker.internal` in Docker**

Current code (lines ~724-731):
```python
    def _try_ollama():
        print("Szukam lokalnej Ollamy (port 11434)...")
        models = detect_ollama()
        if models:
            picked = _pick(models)
            print(f"✓ Lokalna Ollama! Model: {picked}")
            return _make_clients("http://localhost:11434", "ollama", picked)
        return None
```

Replace with:
```python
    def _try_ollama():
        in_docker = _is_docker()
        hosts = ["localhost"]
        if in_docker:
            hosts.append("host.docker.internal")
        for host in hosts:
            label = "port 11434" if host == "localhost" else f"{host}:11434"
            print(f"Szukam Ollamy ({label})...")
            base = f"http://{host}:11434"
            models = detect_ollama(base)
            if models:
                picked = _pick(models)
                print(f"✓ Ollama ({label})! Model: {picked}")
                return _make_clients(base, "ollama", picked)
        return None
```

- [ ] **Step 4: Modify `_try_lecturer` to resolve through Docker host**

When `LECTURER_SERVER` is set to `http://ADRES_SERWERA:PORT` (placeholder), skip as before. But when students inside Docker set `LECTURER_SERVER = "http://192.168.x.x:4242"` (the lecturer's LAN IP), it should also work — and it already does since Docker containers can reach LAN IPs. No change needed for this case.

However, add a convenience: if the student sets `LECTURER_SERVER = "http://host.docker.internal:4242"`, that should work too. It already does since `_try_lecturer` uses the URL as-is. No code change needed here.

Print a Docker hint if nothing is found:
```python
    # At the end of connect_llm, before the final return:
    if _is_docker():
        print("💡 Docker: upewnij się, że LM Studio/Ollama działa na Twoim komputerze (nie w kontenerze).")
        print("   connect_llm automatycznie szuka na host.docker.internal.")
```

- [ ] **Step 5: Test locally (outside Docker)**

Run the notebook cell with `connect_llm` outside Docker to verify no regression:
```bash
cd /Users/kamiljedryczek/Documents/ALK/MachineLearningCodes/MachineLearningCourse
python -c "
from utils import _is_docker, connect_llm
print('Docker:', _is_docker())  # Should be False
# Don't actually connect, just verify import works
print('OK')
"
```

Expected: `Docker: False` then `OK`

- [ ] **Step 6: Test inside Docker**

```bash
docker compose up -d
docker compose exec jupyter python -c "
from utils import _is_docker
print('Docker:', _is_docker())  # Should be True
"
docker compose down
```

Expected: `Docker: True`

- [ ] **Step 7: Commit**

```bash
git add utils.py
git commit -m "$(cat <<'EOF'
connect_llm: auto-detect Docker, try host.docker.internal

When running inside a Docker container, connect_llm now also
probes host.docker.internal for LM Studio and Ollama, so
Docker students can use LLMs running on their host machine
without changing any notebook code.
EOF
)"
```

---

### Task 3: `llm_proxy.py` — add `--tunnel` flag for Cloudflare Quick Tunnel

**Files:**
- Modify: `llm_proxy.py`

Cloudflare Quick Tunnel (`cloudflared tunnel --url http://localhost:PORT`) creates a public HTTPS URL that tunnels to a local port. Zero config, no account needed, free, no bandwidth limits. The proxy's existing Bearer token auth passes through transparently.

The `--tunnel` flag will:
1. Check if `cloudflared` is installed
2. Start it as a subprocess
3. Parse the public URL from its stderr output
4. Print the URL prominently for the lecturer to share with Colab students
5. Kill the subprocess on Ctrl+C

- [ ] **Step 1: Add `--tunnel` CLI argument**

After the existing `parser.add_argument("--verbose"...)` (line ~63), add:

```python
parser.add_argument("--tunnel", "-t", action="store_true",
                    help="Uruchom Cloudflare Quick Tunnel (publiczny URL dla studentów na Colabie)")
```

- [ ] **Step 2: Add tunnel launcher function**

Add this function before `main()` (around line ~762, before the `_get_local_ip` function):

```python
def _start_tunnel(port):
    """Start cloudflared quick tunnel, return (process, public_url)."""
    import shutil
    import subprocess
    import re

    if not shutil.which("cloudflared"):
        print("❌ cloudflared nie jest zainstalowany.")
        print("   Instalacja: brew install cloudflared  (macOS)")
        print("              lub: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
        return None, None

    print(f"🌐 Uruchamiam Cloudflare Tunnel (port {port})...")
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    url = None
    deadline = time.time() + 30
    while time.time() < deadline:
        line = proc.stderr.readline()
        if not line:
            if proc.poll() is not None:
                break
            continue
        match = re.search(r"(https://[a-z0-9-]+\.trycloudflare\.com)", line)
        if match:
            url = match.group(1)
            break

    if not url:
        print("❌ Nie udało się uzyskać URL tunelu. Sprawdź połączenie z internetem.")
        proc.kill()
        return None, None

    print(f"   ✅ Tunel aktywny!")
    return proc, url
```

- [ ] **Step 3: Integrate tunnel into `main()`**

In `main()`, after the proxy and dashboard servers are started, and after the student URL is printed (around line ~833), add the tunnel logic:

```python
    # ── Tunnel (opcjonalny) ──
    tunnel_proc = None
    tunnel_url = None
    if args.tunnel:
        tunnel_proc, tunnel_url = _start_tunnel(args.proxy_port)
        if tunnel_url:
            print(f"\n   🌐 URL dla Colab / zdalnych studentów:")
            print(f"      LECTURER_SERVER = \"{tunnel_url}\"")
            if args.student_key:
                print(f"      Hasło: {'*' * len(args.student_key)}")
            print()
```

- [ ] **Step 4: Clean up tunnel on exit**

In the `KeyboardInterrupt` handler at the end of `main()`, add tunnel cleanup:

After `dashboard_server.shutdown()`, add:
```python
        if tunnel_proc:
            tunnel_proc.kill()
            tunnel_proc.wait()
            print("  Tunel zamknięty.")
```

- [ ] **Step 5: Force student_key when tunnel is active**

Add a safety check in `main()`, right after the auto-detekcja section (line ~811), before starting the proxy:

```python
    # Bezpieczeństwo: tunel wymaga hasła
    if args.tunnel and not args.student_key:
        print("⚠️  --tunnel wymaga --student-key (hasło dla studentów).")
        print("   Bez hasła każdy z internetem mógłby korzystać z Twojego LLM-a!")
        args.student_key = input("   Podaj hasło (np. 'alk-2026'): ").strip()
        if not args.student_key:
            print("❌ Hasło jest wymagane przy tunelowaniu. Uruchom ponownie z -s <hasło>.")
            return
```

- [ ] **Step 6: Test tunnel locally**

```bash
# First install cloudflared if needed
brew install cloudflared  # or appropriate for your system

# Test the tunnel flag
python llm_proxy.py --tunnel -s test-key-2026
```

Expected: proxy starts, cloudflared tunnel starts, public URL is printed. Ctrl+C stops both.

- [ ] **Step 7: Commit**

```bash
git add llm_proxy.py
git commit -m "$(cat <<'EOF'
llm_proxy: add --tunnel flag for Cloudflare Quick Tunnel

Starts cloudflared subprocess to create a public HTTPS URL,
enabling Colab and remote students to reach the proxy.
Requires --student-key for security when tunneling.
EOF
)"
```

---

### Task 4: Update `docs/PLAN_B_DOCKER.md` — LLM access for Docker students

**Files:**
- Modify: `docs/PLAN_B_DOCKER.md`

Docker students need to know: (1) they can use their own LM Studio/Ollama running on the host, and (2) they can connect to the lecturer's proxy. The `connect_llm` function handles this automatically — no code changes in the notebook needed.

- [ ] **Step 1: Add LLM section to PLAN_B_DOCKER.md**

After the "Rozwiązywanie problemów" section (at the end of the file), add:

```markdown

---

## Używanie modeli językowych (LLM) w Dockerze

Na zajęciach z Function Calling i RAG potrzebujesz dostępu do modelu językowego.
Notebook automatycznie szuka LLM-a — nie musisz nic zmieniać w kodzie.

### Opcja 1: Własny LLM na komputerze (zalecana jeśli masz ≥16 GB RAM)

1. Zainstaluj **LM Studio** lub **Ollamę** na swoim komputerze (nie w Dockerze!) — szczegóły w [LOKALNE_LLM.md](LOKALNE_LLM.md)
2. Uruchom model (np. `gemma4:e4b`)
3. Uruchom Docker: `docker compose up`
4. W notebooku uruchom komórkę z `connect_llm` — wykryje LLM automatycznie

> **Jak to działa?** Kontener Docker automatycznie szuka LLM-a na Twoim komputerze
> przez adres `host.docker.internal`. Nie musisz konfigurować sieci.

### Opcja 2: Serwer prowadzącego (na zajęciach)

1. Prowadzący poda adres serwera (np. `http://192.168.1.100:4242`)
2. W notebooku zmień `LECTURER_SERVER`:
   ```python
   LECTURER_SERVER = "http://192.168.1.100:4242"  # ← adres od prowadzącego
   ```
3. Uruchom komórkę — `connect_llm` połączy się z serwerem

### Najczęstsze problemy z LLM w Dockerze

**"connect_llm nic nie znajduje"**

Sprawdź czy LM Studio / Ollama działa na Twoim komputerze (nie w kontenerze):
- LM Studio: otwórz aplikację → upewnij się, że model jest załadowany i serwer uruchomiony
- Ollama: `ollama ps` powinno pokazać załadowany model

**"Model działa bardzo wolno"**

Nie uruchamiaj LLM-a wewnątrz Dockera na Macu — Docker na macOS nie ma dostępu do GPU Apple Silicon.
Zainstaluj LM Studio / Ollamę bezpośrednio na Macu, a Docker połączy się automatycznie.
```

- [ ] **Step 2: Commit**

```bash
git add docs/PLAN_B_DOCKER.md
git commit -m "$(cat <<'EOF'
docs: add LLM access guide for Docker students

Explains how Docker students can use their own LM Studio/Ollama
on the host machine (auto-detected via host.docker.internal)
or connect to the lecturer's proxy.
EOF
)"
```

---

### Task 5: Update `docs/PLAN_C_COLAB.md` — tunnel access for Colab students

**Files:**
- Modify: `docs/PLAN_C_COLAB.md`

Colab students can ONLY use the lecturer's proxy (no local LLM possible). The tunnel URL is shared by the lecturer at the start of class.

- [ ] **Step 1: Add LLM section to PLAN_C_COLAB.md**

After the "Ważne informacje o Colabie" section, before "Alternatywa: MyBinder", add:

```markdown

## Używanie modeli językowych (LLM) w Colabie

Na zajęciach z Function Calling i RAG potrzebujesz dostępu do modelu językowego.
W Colabie używamy **serwera prowadzącego** przez internet.

### Konfiguracja

Prowadzący poda na zajęciach:
- **Adres serwera** (np. `https://abc-xyz.trycloudflare.com`)
- **Hasło** (np. `alk-2026`)

W notebooku zmień `LECTURER_SERVER`:
```python
LECTURER_SERVER = "https://abc-xyz.trycloudflare.com"  # ← adres od prowadzącego
```

Uruchom komórkę z `connect_llm` — poprosi o hasło, a potem połączy się z serwerem.

> **Uwaga:** Adres serwera zmienia się na każdych zajęciach. Zawsze pytaj prowadzącego o aktualny.

### Jeśli nie działa

1. Sprawdź czy wpisałeś `https://` (nie `http://`)
2. Sprawdź czy adres jest aktualny (prowadzący mógł restartować tunel)
3. Sprawdź hasło — prowadzący poda je na zajęciach
```

- [ ] **Step 2: Commit**

```bash
git add docs/PLAN_C_COLAB.md
git commit -m "$(cat <<'EOF'
docs: add LLM access guide for Colab students

Explains how Colab students connect to the lecturer's proxy
via Cloudflare Tunnel URL shared in class.
EOF
)"
```

---

### Task 6: Update `docs/SERWER_PROWADZACEGO.md` — tunnel section

**Files:**
- Modify: `docs/SERWER_PROWADZACEGO.md`

The lecturer needs to know how to expose the proxy for remote/Colab students.

- [ ] **Step 1: Add tunnel section to SERWER_PROWADZACEGO.md**

After the existing "Weryfikacja połączenia z wiersza poleceń" section (at the end), add:

```markdown


## Dostęp zdalny (Google Colab / studenci poza siecią WiFi)

Studenci na Colabie lub poza siecią lokalną mogą połączyć się przez tunel internetowy.
Proxy obsługuje to jednym parametrem:

### Uruchomienie z tunelem

```bash
python llm_proxy.py --tunnel -s alk-2026
```

Proxy wypisze:
```
🌐 URL dla Colab / zdalnych studentów:
   LECTURER_SERVER = "https://abc-xyz.trycloudflare.com"
```

Podaj ten adres i hasło studentom na Colabie.

### Wymagania

- **cloudflared** — instalacja: `brew install cloudflared` (macOS) lub [cloudflare.com](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
- **Hasło (`-s`)** — wymagane przy tunelowaniu (zabezpieczenie przed nieautoryzowanym dostępem)

### Ważne

- Adres tunelu zmienia się przy każdym uruchomieniu — trzeba go podać na nowo
- Istniejące zabezpieczenia (hasło + rate limiting) chronią przed nieautoryzowanym dostępem
- Tunel działa przez HTTPS — certyfikat SSL jest automatyczny
```

- [ ] **Step 2: Commit**

```bash
git add docs/SERWER_PROWADZACEGO.md
git commit -m "$(cat <<'EOF'
docs: add tunnel section for remote/Colab students

Documents the --tunnel flag in llm_proxy.py for exposing the
proxy via Cloudflare Quick Tunnel.
EOF
)"
```

---

### Task 7: Update `docs/LOKALNE_LLM.md` — Docker note

**Files:**
- Modify: `docs/LOKALNE_LLM.md`

Add a short Docker-specific note so Docker students don't try to install LM Studio inside the container.

- [ ] **Step 1: Add Docker note**

After the "Podsumowanie — co wybrać?" section (at the very end), add:

```markdown

---

## Docker (Plan B) — ważna uwaga

Jeśli korzystasz z Dockera (Plan B), zainstaluj LM Studio / Ollamę **na swoim komputerze**, nie w kontenerze Docker. Notebook automatycznie wykryje LLM na hoście.

> **Dlaczego nie w Dockerze?** Na Macu z Apple Silicon Docker nie ma dostępu do GPU — model działałby 10-15x wolniej. Na Windowsie/Linuksie z kartą NVIDIA jest to możliwe, ale niepotrzebnie komplikuje konfigurację.
```

- [ ] **Step 2: Commit**

```bash
git add docs/LOKALNE_LLM.md
git commit -m "$(cat <<'EOF'
docs: add Docker note about installing LLM on host, not container
EOF
)"
```

---

### Task 8: End-to-end verification

- [ ] **Step 1: Verify Docker student flow (host LLM)**

Start LM Studio on the host, then test from Docker:
```bash
docker compose up -d
docker compose exec jupyter python -c "
from utils import connect_llm
client, instr, model = connect_llm()
if client:
    print(f'SUCCESS: {model}')
else:
    print('FAIL: no LLM found')
"
docker compose down
```

Expected: `SUCCESS: gemma-...` (found via `host.docker.internal`)

- [ ] **Step 2: Verify Docker student flow (lecturer proxy)**

Start `llm_proxy.py` on the host, then test from Docker:
```bash
# Terminal 1: start proxy
python llm_proxy.py -s test-key

# Terminal 2: test from Docker
docker compose up -d
docker compose exec jupyter python -c "
from utils import connect_llm
client, instr, model = connect_llm(lecturer_server='http://host.docker.internal:4242', api_key='test-key')
if client:
    print(f'SUCCESS: {model}')
else:
    print('FAIL')
"
docker compose down
```

Expected: `SUCCESS: <model>`

- [ ] **Step 3: Verify tunnel flow**

```bash
python llm_proxy.py --tunnel -s test-key-2026
# Note the https://xxx.trycloudflare.com URL

# In another terminal, test with curl:
curl -s -H "Authorization: Bearer test-key-2026" https://xxx.trycloudflare.com/v1/models
```

Expected: JSON with model list

- [ ] **Step 4: Final commit (if any fixes needed)**

```bash
git add -A
git status  # verify only expected files
git commit -m "fix: address issues found during e2e testing"
```
