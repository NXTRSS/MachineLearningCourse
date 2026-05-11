#!/usr/bin/env python3
"""
LLM Proxy — transparentny proxy między studentami a LM Studio.

Uruchomienie:
    python llm_proxy.py                                     # proxy :4242 → LM Studio :4141
    python llm_proxy.py --proxy-port 5000 --lm-port 1234    # inne porty
    python llm_proxy.py --api-key sk-lm-xxx -s alk-2026     # ⭐ studenci używają krótkiego hasła

Bezpieczne podawanie kluczy (priorytet: CLI → env var → .env → prompt):
    1. Plik .env (najwygodniejsze — raz ustawisz, zapominasz):
         echo 'LLM_API_KEY=sk-dlugi-klucz'  >> .env
         echo 'LLM_STUDENT_KEY=alk-2026'     >> .env
         python llm_proxy.py                              # czyta z .env

    2. Zmienne środowiskowe:
         export LLM_API_KEY=sk-dlugi-klucz
         python llm_proxy.py -s alk-2026

    3. Interaktywny prompt (gdy student_key jest ustawiony ale api_key nie):
         python llm_proxy.py -s alk-2026     # zapyta o API key (ukryte znaki)

Studenci łączą się na port proxy (4242) — reszta bez zmian.
Prowadzący widzi:
  - Terminal: live dashboard (kto czeka, kto jest obsługiwany, czasy)
  - Przeglądarka: http://localhost:5050 — web dashboard z wykresami

⚠️  Ten plik + .env są w .gitignore — nigdy nie trafiają na GitHub.
Ctrl+C aby zakończyć.
"""

import argparse
import json
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import requests

# ── CLI ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="LLM Proxy z monitoringiem")
parser.add_argument("--proxy-port", type=int, default=4242,
                    help="Port na którym studenci się łączą (domyślnie 4242)")
parser.add_argument("--lm-port", type=int, default=4141,
                    help="Port LM Studio (domyślnie 4141)")
parser.add_argument("--lm-host", type=str, default="localhost",
                    help="Host LM Studio (domyślnie localhost)")
parser.add_argument("--dashboard-port", type=int, default=5050,
                    help="Port web dashboardu (domyślnie 5050)")
parser.add_argument("--api-key", "-k", type=str, default=None,
                    help="API key do LM Studio (proxy forwarduje go dalej)")
parser.add_argument("--student-key", "-s", type=str, default=None,
                    help="Krótkie hasło dla studentów (np. 'alk-2026'). "
                         "Jeśli ustawione, proxy wymaga go od klientów i podmienia na --api-key.")
parser.add_argument("--env-file", type=str, default=".env",
                    help="Ścieżka do pliku z kluczami (domyślnie .env)")
parser.add_argument("--verbose", "-v", action="store_true",
                    help="Live dashboard w terminalu (domyślnie: cichy tryb z logami)")
args = parser.parse_args()


# ── Resolve secrets: CLI arg → env var → .env file → interactive prompt ──

import getpass
import os

def _load_env_file(path):
    """Wczytaj klucze z pliku .env (KEY=VALUE, po jednym na linię)."""
    values = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip("'\"")
    return values

from pathlib import Path
_env = _load_env_file(args.env_file)

def _resolve_secret(cli_value, env_name, prompt_text):
    """CLI arg → zmienna środowiskowa → .env → interaktywny prompt."""
    if cli_value:
        return cli_value
    if os.environ.get(env_name):
        return os.environ[env_name]
    if _env.get(env_name):
        return _env[env_name]
    return None  # nie pytaj interaktywnie — oba klucze opcjonalne

args.api_key = _resolve_secret(args.api_key, "LLM_API_KEY", "API key do LM Studio")
args.student_key = _resolve_secret(args.student_key, "LLM_STUDENT_KEY", "Hasło dla studentów")

# Jeśli student_key jest ustawiony ale api_key nie — zapytaj interaktywnie
if args.student_key and not args.api_key:
    args.api_key = getpass.getpass("🔑 API key do LM Studio (ukryte): ")

LM_BASE = f"http://{args.lm_host}:{args.lm_port}"


# ═══════════════════════════════════════════════════════════════════════
#  STATS — współdzielony stan
# ═══════════════════════════════════════════════════════════════════════

class RequestInfo:
    """Jedno zapytanie studenta."""
    def __init__(self, req_id, client_ip, endpoint, model="?", question="", username=""):
        self.req_id = req_id
        self.client_ip = client_ip
        self.endpoint = endpoint
        self.model = model
        self.question = question[:80]
        self.username = username
        self.start_time = time.time()
        self.end_time = None
        self.status = "active"  # active / done / error
        self.queue_position = 0

    @property
    def duration(self):
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self):
        return {
            "id": self.req_id[:8],
            "ip": self.client_ip,
            "username": self.username,
            "endpoint": self.endpoint,
            "model": self.model,
            "question": self.question,
            "duration": round(self.duration, 1),
            "status": self.status,
            "start": datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S"),
        }


class ProxyStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.active: dict[str, RequestInfo] = {}
        self.history: deque[RequestInfo] = deque(maxlen=200)
        self.total = 0
        self.errors = 0
        self.start_time = time.time()

    def new_request(self, client_ip, endpoint, model="?", question="", username="") -> RequestInfo:
        req_id = str(uuid.uuid4())
        info = RequestInfo(req_id, client_ip, endpoint, model, question, username)
        with self.lock:
            info.queue_position = len(self.active) + 1
            self.active[req_id] = info
        return info

    def end_request(self, info: RequestInfo, error=False):
        info.end_time = time.time()
        info.status = "error" if error else "done"
        with self.lock:
            self.active.pop(info.req_id, None)
            self.history.appendleft(info)
            self.total += 1
            if error:
                self.errors += 1

    @property
    def active_count(self):
        return len(self.active)

    def requests_last_60s(self):
        now = time.time()
        return sum(1 for r in self.history if now - r.start_time < 60)

    def avg_time_last_60s(self):
        now = time.time()
        recent = [r for r in self.history if now - r.start_time < 60 and r.end_time]
        if not recent:
            return 0.0
        return sum(r.duration for r in recent) / len(recent)

    def avg_time_all(self):
        done = [r for r in self.history if r.end_time]
        if not done:
            return 0.0
        return sum(r.duration for r in done) / len(done)

    def max_time(self):
        done = [r for r in self.history if r.end_time]
        return max((r.duration for r in done), default=0.0)

    def unique_clients(self):
        ips = set(r.client_ip for r in self.history)
        ips.update(r.client_ip for r in self.active.values())
        return len(ips)

    def active_clients(self):
        return set(r.client_ip for r in self.active.values())

    def unique_usernames(self):
        names = set()
        for r in self.history:
            if r.username:
                names.add(r.username)
        for r in self.active.values():
            if r.username:
                names.add(r.username)
        return sorted(names)

    def clear_history(self):
        with self.lock:
            self.history.clear()
            self.total = 0
            self.errors = 0

    def snapshot(self, page=0, per_page=20):
        """Snapshot danych do web dashboardu z paginacją."""
        with self.lock:
            hist = list(self.history)
            start = page * per_page
            page_items = hist[start : start + per_page]
            return {
                "active": [r.to_dict() for r in self.active.values()],
                "history": [r.to_dict() for r in page_items],
                "history_total": len(hist),
                "history_page": page,
                "history_pages": max(1, (len(hist) + per_page - 1) // per_page),
                "total": self.total,
                "errors": self.errors,
                "active_count": self.active_count,
                "unique_clients": self.unique_clients(),
                "unique_usernames": self.unique_usernames(),
                "avg_time_60s": round(self.avg_time_last_60s(), 1),
                "avg_time_all": round(self.avg_time_all(), 1),
                "max_time": round(self.max_time(), 1),
                "requests_60s": self.requests_last_60s(),
                "uptime": int(time.time() - self.start_time),
            }


stats = ProxyStats()


# ═══════════════════════════════════════════════════════════════════════
#  RATE LIMITER — blokuje brute force na hasło
# ═══════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Po MAX_FAILS nieudanych prób z danego IP → blokada na BLOCK_SECONDS."""

    MAX_FAILS = 5
    BLOCK_SECONDS = 60

    def __init__(self):
        self._lock = threading.Lock()
        self._fails: dict[str, list[float]] = {}   # ip → [timestamps]
        self._blocked: dict[str, float] = {}        # ip → unblock_time

    def is_blocked(self, ip: str) -> bool:
        with self._lock:
            unblock = self._blocked.get(ip, 0)
            if time.time() < unblock:
                return True
            if ip in self._blocked:
                del self._blocked[ip]
            return False

    def record_fail(self, ip: str):
        now = time.time()
        with self._lock:
            times = self._fails.setdefault(ip, [])
            times.append(now)
            # Licz tylko ostatnie 60s
            times[:] = [t for t in times if now - t < 60]
            if len(times) >= self.MAX_FAILS:
                self._blocked[ip] = now + self.BLOCK_SECONDS
                self._fails.pop(ip, None)

    def record_success(self, ip: str):
        with self._lock:
            self._fails.pop(ip, None)


rate_limiter = RateLimiter()


# ═══════════════════════════════════════════════════════════════════════
#  PROXY — przechwytuje zapytania i forwarduje do LM Studio
# ═══════════════════════════════════════════════════════════════════════

class ProxyHandler(BaseHTTPRequestHandler):
    """Proxy HTTP: student → proxy → LM Studio → proxy → student."""

    def log_message(self, format, *a):
        pass  # ciszej — dashboard wyświetla co trzeba

    def _send_json_error(self, code, message):
        """Wyślij błąd jako JSON (kompatybilny z klientem OpenAI)."""
        body = json.dumps({"error": {"message": message, "type": "auth_error", "code": code}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _forward(self, method):
        client_ip = self.client_address[0]
        path = self.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # ── /v1/models — przepuszczaj bez auth (health check) ──
        endpoint = path.split("?")[0]
        skip_auth = endpoint.rstrip("/") in ("/v1/models", "/v1/models/")

        if args.student_key and not skip_auth:
            # ── Rate limit: blokada po zbyt wielu nieudanych próbach ──
            if rate_limiter.is_blocked(client_ip):
                self._send_json_error(429, "Zbyt wiele nieudanych prób. Spróbuj za minutę.")
                return

            # ── Auth: sprawdź hasło studenta ──
            auth = self.headers.get("Authorization", "")
            token = auth.replace("Bearer ", "").strip() if auth.startswith("Bearer ") else ""
            if token != args.student_key:
                rate_limiter.record_fail(client_ip)
                self._send_json_error(401, "Nieprawidłowy klucz. Podaj prawidłowy --api-key przy łączeniu.")
                return
            rate_limiter.record_success(client_ip)

        # Wyciągnij info z body (model, pytanie)
        model = "?"
        question = ""
        endpoint = path.split("?")[0]
        if body:
            try:
                data = json.loads(body)
                model = data.get("model", "?")
                msgs = data.get("messages", [])
                if msgs:
                    last_user = [m for m in msgs if m.get("role") == "user"]
                    if last_user:
                        question = last_user[-1].get("content", "")[:80]
            except (json.JSONDecodeError, AttributeError):
                pass

        # Nazwa studenta z nagłówka (wysyłana przez chat_ui)
        student_name = self.headers.get("X-Student-Name", "")

        # Rejestruj zapytanie (pomijaj ping-like requesty jak /v1/models)
        is_chat = "/chat/completions" in path or "/completions" in path
        info = None
        if is_chat:
            info = stats.new_request(client_ip, endpoint, model, question, student_name)

        # Forwarduj do LM Studio
        target_url = f"{LM_BASE}{path}"
        fwd_headers = {}
        for key, val in self.headers.items():
            if key.lower() not in ("host", "content-length", "transfer-encoding"):
                fwd_headers[key] = val
        # Dodaj API key jeśli skonfigurowany
        if args.api_key:
            fwd_headers["Authorization"] = f"Bearer {args.api_key}"

        try:
            resp = requests.request(
                method=method,
                url=target_url,
                headers=fwd_headers,
                data=body,
                timeout=300,
                stream=False,
            )
            if info:
                stats.end_request(info, error=(resp.status_code >= 400))

            # Odeślij odpowiedź do studenta
            self.send_response(resp.status_code)
            for key, val in resp.headers.items():
                if key.lower() not in ("transfer-encoding", "connection", "content-encoding", "content-length"):
                    self.send_header(key, val)
            response_body = resp.content
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        except requests.ConnectionError:
            if info:
                stats.end_request(info, error=True)
            self.send_error(502, "LM Studio niedostępne")

        except requests.Timeout:
            if info:
                stats.end_request(info, error=True)
            self.send_error(504, "LM Studio timeout (>300s)")

        except Exception as e:
            if info:
                stats.end_request(info, error=True)
            self.send_error(500, f"Proxy error: {e}")

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()


# ═══════════════════════════════════════════════════════════════════════
#  WEB DASHBOARD — http://localhost:5050
# ═══════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8">
<title>LLM Proxy Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 20px; }
  h1 { color: #58a6ff; margin-bottom: 4px; font-size: 1.5em; }
  .subtitle { color: #8b949e; margin-bottom: 20px; font-size: 0.9em; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center; }
  .card .value { font-size: 2em; font-weight: bold; color: #58a6ff; }
  .card .label { color: #8b949e; font-size: 0.85em; margin-top: 4px; }
  .card.warn .value { color: #d29922; }
  .card.error .value { color: #f85149; }
  .card.ok .value { color: #3fb950; }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; }
  th { background: #161b22; color: #8b949e; text-align: left; padding: 8px 12px; font-size: 0.85em;
       border-bottom: 1px solid #30363d; }
  td { padding: 8px 12px; border-bottom: 1px solid #21262d; font-size: 0.9em; }
  tr:hover { background: #161b22; }
  .status-active { color: #d29922; font-weight: bold; }
  .status-done { color: #3fb950; }
  .status-error { color: #f85149; }
  .section { margin-top: 24px; }
  .section h2 { color: #c9d1d9; font-size: 1.1em; margin-bottom: 8px;
                 border-bottom: 1px solid #30363d; padding-bottom: 6px; }
  .bar-chart { display: flex; align-items: flex-end; gap: 3px; height: 60px; margin-top: 8px; }
  .bar { background: #58a6ff; border-radius: 2px 2px 0 0; min-width: 8px; flex: 1;
         transition: height 0.3s; }
  .bar.empty { background: #21262d; }
  .bar-labels { display: flex; gap: 3px; margin-top: 2px; }
  .bar-labels span { flex: 1; text-align: center; font-size: 0.65em; color: #484f58; }
  .pulse { animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
  .question { max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #8b949e; }
</style>
</head>
<body>
<h1>🖥️ LLM Proxy Dashboard</h1>
<p class="subtitle">PROXY_PORT → LM Studio :LM_PORT &nbsp;·&nbsp; <span id="clock"></span></p>
<p style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px; margin:12px 0; font-size:1.05em;">
  📋 Adres dla studentów: &nbsp;<code style="color:#58a6ff; font-size:1.1em;">STUDENT_URL</code>
</p>

<div class="cards" id="cards"></div>

<div class="section">
  <h2>👥 Zalogowani studenci</h2>
  <p id="student-list" style="color:#58a6ff; font-size:1.1em; margin-top:6px;">—</p>
</div>

<div class="section">
  <h2>📊 Zapytania / minutę (ostatnie 10 min)</h2>
  <div class="bar-chart" id="chart"></div>
  <div class="bar-labels" id="chart-labels"></div>
</div>

<div class="section">
  <h2 id="active-title">⏳ Aktywne zapytania</h2>
  <table id="active-table">
    <thead><tr><th>Kto</th><th>Czas</th><th>Model</th><th>Pytanie</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<div class="section">
  <div style="display:flex; align-items:center; gap:12px; border-bottom:1px solid #30363d; padding-bottom:6px; margin-bottom:8px;">
    <h2 style="margin:0; border:none; padding:0;">📋 Ostatnie zapytania</h2>
    <button onclick="clearHistory()" style="background:#21262d; color:#c9d1d9; border:1px solid #30363d;
      border-radius:6px; padding:4px 12px; cursor:pointer; font-size:0.85em;">🗑️ Wyczyść</button>
    <span style="margin-left:auto; color:#8b949e; font-size:0.85em;" id="page-info"></span>
    <button onclick="changePage(-1)" id="btn-prev" style="background:#21262d; color:#c9d1d9; border:1px solid #30363d;
      border-radius:6px; padding:4px 8px; cursor:pointer;">◀</button>
    <button onclick="changePage(1)" id="btn-next" style="background:#21262d; color:#c9d1d9; border:1px solid #30363d;
      border-radius:6px; padding:4px 8px; cursor:pointer;">▶</button>
  </div>
  <table id="history-table">
    <thead><tr><th>Czas</th><th>Kto</th><th>Czas trwania</th><th>Status</th><th>Pytanie</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<script>
const REFRESH = 2000;
let chartData = new Array(20).fill(0);
let lastTotal = null;
let currentPage = 0;

function fmt(s) { return s < 1 ? Math.round(s*1000)+'ms' : s.toFixed(1)+'s'; }

async function clearHistory() {
  if (!confirm('Wyczyścić historię zapytań?')) return;
  await fetch('/api/clear', {method:'POST'});
  currentPage = 0;
  refresh();
}

function changePage(delta) {
  currentPage = Math.max(0, currentPage + delta);
  refresh();
}

async function refresh() {
  try {
    const r = await fetch('/api/stats?page=' + currentPage);
    const d = await r.json();
    if (currentPage >= d.history_pages) currentPage = Math.max(0, d.history_pages - 1);
    document.getElementById('clock').textContent = new Date().toLocaleTimeString('pl');

    // Cards
    const activeClass = d.active_count > 0 ? (d.active_count > 3 ? 'error' : 'warn') : 'ok';
    document.getElementById('cards').innerHTML = `
      <div class="card ${activeClass}">
        <div class="value">${d.active_count}</div><div class="label">Aktywne</div></div>
      <div class="card ok"><div class="value">${(d.unique_usernames||[]).length || d.unique_clients}</div><div class="label">Studenci</div></div>
      <div class="card"><div class="value">${d.total}</div><div class="label">Łącznie</div></div>
      <div class="card ${d.errors > 0 ? 'error' : ''}">
        <div class="value">${d.errors}</div><div class="label">Błędy</div></div>
      <div class="card"><div class="value">${fmt(d.avg_time_60s)}</div><div class="label">Śr. czas (1 min)</div></div>
      <div class="card"><div class="value">${fmt(d.max_time)}</div><div class="label">Najdłuższy</div></div>
    `;

    // Chart — track requests per refresh interval
    if (lastTotal !== null) {
      chartData.push(d.total - lastTotal);
      chartData.shift();
    }
    lastTotal = d.total;
    const maxBar = Math.max(1, ...chartData);
    document.getElementById('chart').innerHTML = chartData.map(v =>
      `<div class="bar ${v===0?'empty':''}" style="height:${Math.max(2, v/maxBar*100)}%"></div>`
    ).join('');

    // Active table
    document.getElementById('active-title').innerHTML =
      d.active_count > 0
        ? `⏳ Aktywne zapytania <span class="pulse" style="color:#d29922">(${d.active_count})</span>`
        : '⏳ Aktywne zapytania (0)';
    const aBody = document.querySelector('#active-table tbody');
    aBody.innerHTML = d.active.map(r => `<tr>
      <td>${r.username || r.ip}</td>
      <td class="status-active">${fmt(r.duration)}...</td>
      <td>${r.model}</td>
      <td class="question">${r.question || '—'}</td>
    </tr>`).join('') || '<tr><td colspan="4" style="color:#484f58">Brak aktywnych zapytań</td></tr>';

    // History table
    // Student list
    const names = d.unique_usernames || [];
    const nameEl = document.getElementById('student-list');
    if (nameEl) nameEl.textContent = names.length ? names.join(', ') : '—';

    const hBody = document.querySelector('#history-table tbody');
    hBody.innerHTML = d.history.map(r => `<tr>
      <td>${r.start}</td>
      <td>${r.username || r.ip}</td>
      <td>${fmt(r.duration)}</td>
      <td class="status-${r.status}">${r.status === 'done' ? '✓' : '✗'}</td>
      <td class="question">${r.question || '—'}</td>
    </tr>`).join('') || '<tr><td colspan="5" style="color:#484f58">Brak zapytań</td></tr>';

    // Paginacja
    document.getElementById('page-info').textContent =
      d.history_total > 0 ? `${d.history_page + 1} / ${d.history_pages}  (${d.history_total} zapytań)` : '';
    document.getElementById('btn-prev').disabled = d.history_page <= 0;
    document.getElementById('btn-next').disabled = d.history_page >= d.history_pages - 1;

  } catch(e) { console.error(e); }
}

setInterval(refresh, REFRESH);
refresh();
</script>
</body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """Web dashboard — serwuje HTML + JSON API."""

    def log_message(self, format, *a):
        pass

    def do_POST(self):
        if self.path == "/api/clear":
            stats.clear_history()
            body = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path.startswith("/api/stats"):
            # Paginacja: /api/stats?page=0
            page = 0
            if "page=" in self.path:
                try:
                    page = int(self.path.split("page=")[1].split("&")[0])
                except ValueError:
                    pass
            data = json.dumps(stats.snapshot(page=page)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            html = DASHBOARD_HTML.replace("PROXY_PORT", str(args.proxy_port))
            html = html.replace("LM_PORT", str(args.lm_port))
            html = html.replace("STUDENT_URL", f"http://{_get_local_ip()}:{args.proxy_port}")
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


# ═══════════════════════════════════════════════════════════════════════
#  TERMINAL DASHBOARD — live w terminalu
# ═══════════════════════════════════════════════════════════════════════

CLEAR = "\033[2J\033[H"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"


def format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.1f}s"


def render_terminal():
    """Renderuj dashboard w terminalu."""
    now = datetime.now().strftime("%H:%M:%S")
    snap = stats.snapshot()

    lines = []
    _lip = _get_local_ip()
    _surl = f"http://{_lip}:{args.proxy_port}"
    lines.append(f"{BOLD}╔═══════════════════════════════════════════════════════════════════╗{RESET}")
    lines.append(f"{BOLD}║  🖥️  LLM Proxy  :{args.proxy_port} → LM Studio :{args.lm_port}              {DIM}{now}{RESET}{BOLD}  ║{RESET}")
    lines.append(f"{BOLD}║  📋 Studenci:   {CYAN}{_surl}{RESET}{BOLD}                          ║{RESET}")
    lines.append(f"{BOLD}║  📊 Dashboard:  http://localhost:{args.dashboard_port}                          {BOLD}║{RESET}")
    lines.append(f"{BOLD}╠═══════════════════════════════════════════════════════════════════╣{RESET}")

    # Metryki
    ac = snap["active_count"]
    ac_color = RED if ac > 3 else YELLOW if ac > 0 else GREEN
    lines.append(f"  Aktywne:       {ac_color}{ac}{RESET}     "
                 f"Łącznie: {snap['total']}     "
                 f"Błędy: {RED if snap['errors'] else ''}{snap['errors']}{RESET}     "
                 f"Klienci: {snap['unique_clients']}")
    lines.append(f"  Śr. czas:      {format_time(snap['avg_time_60s'])} (1min)  /  "
                 f"{format_time(snap['avg_time_all'])} (total)  /  "
                 f"max: {format_time(snap['max_time'])}")
    lines.append(f"  Req/min:       {snap['requests_60s']}")
    usernames = snap.get("unique_usernames", [])
    if usernames:
        lines.append(f"  👥 Studenci:    {CYAN}{', '.join(usernames)}{RESET}")

    # Aktywne zapytania
    lines.append(f"{BOLD}╠═══════════════════════════════════════════════════════════════╣{RESET}")
    if snap["active"]:
        lines.append(f"  {YELLOW}⏳ AKTYWNE ({len(snap['active'])}){RESET}")
        for r in snap["active"]:
            q = r["question"][:40] or "—"
            who = f"{CYAN}{r['username']}{RESET}" if r.get("username") else r["ip"]
            lines.append(f"    {who:<26s} {YELLOW}{format_time(r['duration']):>6s}{RESET}  {q}")
    else:
        lines.append(f"  {GREEN}✓ Brak aktywnych zapytań{RESET}")

    # Ostatnie zapytania
    lines.append(f"{BOLD}╠═══════════════════════════════════════════════════════════════╣{RESET}")
    lines.append(f"  {DIM}Ostatnie zapytania:{RESET}")
    for r in snap["history"][:8]:
        status = f"{GREEN}✓{RESET}" if r["status"] == "done" else f"{RED}✗{RESET}"
        q = r["question"][:35] or "—"
        who = f"{CYAN}{r['username']}{RESET}" if r.get("username") else r["ip"]
        lines.append(f"    {r['start']}  {who:<26s} {format_time(r['duration']):>6s}  {status}  {q}")
    if not snap["history"]:
        lines.append(f"    {DIM}(brak){RESET}")

    lines.append(f"{BOLD}╚═══════════════════════════════════════════════════════════════╝{RESET}")
    lines.append(f"  {DIM}Ctrl+C aby zakończyć{RESET}")

    print(CLEAR + "\n".join(lines))


def terminal_loop():
    """Odświeżaj terminal co 2s."""
    while True:
        try:
            render_terminal()
            time.sleep(2)
        except Exception:
            time.sleep(2)


def quiet_loop():
    """Tryb cichy — loguj tylko nowe zapytania (jednoliniowo)."""
    seen = 0
    while True:
        try:
            snap = stats.snapshot()
            history = snap["history"]
            new_count = snap["total"]
            if new_count > seen:
                for r in reversed(history[: new_count - seen]):
                    who = r.get("username") or r["ip"]
                    status = "✓" if r["status"] == "done" else "✗"
                    q = r["question"][:50] or "—"
                    print(f"  {r['start']}  {status}  {who:<16s}  {format_time(r['duration']):>6s}  {q}")
                seen = new_count
            time.sleep(2)
        except Exception:
            time.sleep(2)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def _get_local_ip():
    """Wykryj IP w sieci lokalnej (Wi-Fi / Ethernet)."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # nie wysyła danych, tylko wykrywa interfejs
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def main():
    # Sprawdź czy LM Studio odpowiada
    print(f"🔍 Sprawdzam LM Studio na {LM_BASE}...")
    lm_needs_auth = False
    try:
        # Najpierw bez klucza — sprawdź czy LM Studio wymaga auth
        r = requests.get(f"{LM_BASE}/v1/models", timeout=3)
        if r.status_code == 200:
            models = [m["id"] for m in r.json().get("data", [])]
            print(f"   ✅ LM Studio online — modele: {', '.join(models)}")
        elif r.status_code == 401:
            lm_needs_auth = True
            # Spróbuj z kluczem
            if not args.api_key:
                print(f"   ⚠️  LM Studio wymaga auth — podaj --api-key lub ustaw LLM_API_KEY w .env")
                return
            r2 = requests.get(f"{LM_BASE}/v1/models", timeout=3,
                              headers={"Authorization": f"Bearer {args.api_key}"})
            if r2.status_code == 200:
                models = [m["id"] for m in r2.json().get("data", [])]
                print(f"   ✅ LM Studio online (auth) — modele: {', '.join(models)}")
            else:
                print(f"   ❌ LM Studio odrzuca klucz API (HTTP {r2.status_code})")
                return
        else:
            print(f"   ⚠️  LM Studio odpowiada HTTP {r.status_code}")
    except requests.ConnectionError:
        print(f"   ❌ LM Studio nie odpowiada na {LM_BASE}")
        print(f"      Uruchom LM Studio i spróbuj ponownie.")
        return

    # Auto-detekcja: jeśli LM Studio nie wymaga auth → wyłącz hasło studenta
    if not lm_needs_auth and args.student_key:
        print(f"   ℹ️  LM Studio nie wymaga auth → hasło studenta wyłączone")
        args.student_key = None

    # Uruchom proxy
    proxy_server = HTTPServer(("0.0.0.0", args.proxy_port), ProxyHandler)
    proxy_thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
    proxy_thread.start()
    print(f"   🔀 Proxy nasłuchuje na :{args.proxy_port}")

    # Uruchom dashboard
    dashboard_server = HTTPServer(("0.0.0.0", args.dashboard_port), DashboardHandler)
    dashboard_thread = threading.Thread(target=dashboard_server.serve_forever, daemon=True)
    dashboard_thread.start()
    print(f"   📊 Dashboard: http://localhost:{args.dashboard_port}")

    _local_ip = _get_local_ip()
    _student_url = f"http://{_local_ip}:{args.proxy_port}"
    print(f"\n   📋 Adres dla studentów (do notebooka):")
    print(f"      LECTURER_SERVER = \"{_student_url}\"")
    if args.student_key:
        print(f"   🔑 Hasło dla studentów: ustawione ({'*' * len(args.student_key)})")
    else:
        print(f"   🔓 Bez hasła — otwarty dostęp")
    print(f"   Ctrl+C aby zakończyć\n")

    # Terminal: lekki tryb logów (domyślny) albo live dashboard (-v)
    if args.verbose:
        time.sleep(2)
    else:
        print(f"   {DIM}Logi poniżej · dashboard na http://localhost:{args.dashboard_port} · -v dla live dashboard{RESET}\n")

    try:
        terminal_loop() if args.verbose else quiet_loop()
    except KeyboardInterrupt:
        print(f"\n\n{DIM}Proxy zatrzymane.{RESET}")
        if stats.total > 0:
            print(f"  Łącznie: {stats.total} zapytań, "
                  f"klienci: {stats.unique_clients()}, "
                  f"średni czas: {format_time(stats.avg_time_all())}, "
                  f"błędy: {stats.errors}")
        proxy_server.shutdown()
        dashboard_server.shutdown()


if __name__ == "__main__":
    main()
