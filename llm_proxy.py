#!/usr/bin/env python3
"""
LLM Proxy — transparentny proxy między studentami a LM Studio.

Uruchomienie:
    python llm_proxy.py                                     # proxy :5000 → LM Studio :4242
    python llm_proxy.py --proxy-port 6000 --lm-port 1234    # inne porty
    python llm_proxy.py --api-key sk-lm-xxx                 # z auth do LM Studio

Studenci łączą się na port proxy (5000) — reszta bez zmian.
Prowadzący widzi:
  - Terminal: live dashboard (kto czeka, kto jest obsługiwany, czasy)
  - Przeglądarka: http://localhost:5050 — web dashboard z wykresami

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
parser.add_argument("--proxy-port", type=int, default=5000,
                    help="Port na którym studenci się łączą (domyślnie 5000)")
parser.add_argument("--lm-port", type=int, default=4242,
                    help="Port LM Studio (domyślnie 4242)")
parser.add_argument("--lm-host", type=str, default="localhost",
                    help="Host LM Studio (domyślnie localhost)")
parser.add_argument("--dashboard-port", type=int, default=5050,
                    help="Port web dashboardu (domyślnie 5050)")
parser.add_argument("--api-key", "-k", type=str, default=None,
                    help="API key do LM Studio (proxy forwarduje go dalej)")
args = parser.parse_args()

LM_BASE = f"http://{args.lm_host}:{args.lm_port}"


# ═══════════════════════════════════════════════════════════════════════
#  STATS — współdzielony stan
# ═══════════════════════════════════════════════════════════════════════

class RequestInfo:
    """Jedno zapytanie studenta."""
    def __init__(self, req_id, client_ip, endpoint, model="?", question=""):
        self.req_id = req_id
        self.client_ip = client_ip
        self.endpoint = endpoint
        self.model = model
        self.question = question[:80]
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

    def new_request(self, client_ip, endpoint, model="?", question="") -> RequestInfo:
        req_id = str(uuid.uuid4())
        info = RequestInfo(req_id, client_ip, endpoint, model, question)
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

    def snapshot(self):
        """Snapshot danych do web dashboardu."""
        with self.lock:
            return {
                "active": [r.to_dict() for r in self.active.values()],
                "history": [r.to_dict() for r in list(self.history)[:30]],
                "total": self.total,
                "errors": self.errors,
                "active_count": self.active_count,
                "unique_clients": self.unique_clients(),
                "avg_time_60s": round(self.avg_time_last_60s(), 1),
                "avg_time_all": round(self.avg_time_all(), 1),
                "max_time": round(self.max_time(), 1),
                "requests_60s": self.requests_last_60s(),
                "uptime": int(time.time() - self.start_time),
            }


stats = ProxyStats()


# ═══════════════════════════════════════════════════════════════════════
#  PROXY — przechwytuje zapytania i forwarduje do LM Studio
# ═══════════════════════════════════════════════════════════════════════

class ProxyHandler(BaseHTTPRequestHandler):
    """Proxy HTTP: student → proxy → LM Studio → proxy → student."""

    def log_message(self, format, *a):
        pass  # ciszej — dashboard wyświetla co trzeba

    def _forward(self, method):
        client_ip = self.client_address[0]
        path = self.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

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

        # Rejestruj zapytanie (pomijaj ping-like requesty jak /v1/models)
        is_chat = "/chat/completions" in path or "/completions" in path
        info = None
        if is_chat:
            info = stats.new_request(client_ip, endpoint, model, question)

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

<div class="cards" id="cards"></div>

<div class="section">
  <h2>📊 Zapytania / minutę (ostatnie 10 min)</h2>
  <div class="bar-chart" id="chart"></div>
  <div class="bar-labels" id="chart-labels"></div>
</div>

<div class="section">
  <h2 id="active-title">⏳ Aktywne zapytania</h2>
  <table id="active-table">
    <thead><tr><th>IP</th><th>Czas</th><th>Model</th><th>Pytanie</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<div class="section">
  <h2>📋 Ostatnie zapytania</h2>
  <table id="history-table">
    <thead><tr><th>Czas</th><th>IP</th><th>Czas trwania</th><th>Status</th><th>Pytanie</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<script>
const REFRESH = 2000;
let chartData = new Array(20).fill(0);
let lastTotal = null;

function fmt(s) { return s < 1 ? Math.round(s*1000)+'ms' : s.toFixed(1)+'s'; }

async function refresh() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();
    document.getElementById('clock').textContent = new Date().toLocaleTimeString('pl');

    // Cards
    const activeClass = d.active_count > 0 ? (d.active_count > 3 ? 'error' : 'warn') : 'ok';
    document.getElementById('cards').innerHTML = `
      <div class="card ${activeClass}">
        <div class="value">${d.active_count}</div><div class="label">Aktywne</div></div>
      <div class="card"><div class="value">${d.unique_clients}</div><div class="label">Unikalni klienci</div></div>
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
      <td>${r.ip}</td>
      <td class="status-active">${fmt(r.duration)}...</td>
      <td>${r.model}</td>
      <td class="question">${r.question || '—'}</td>
    </tr>`).join('') || '<tr><td colspan="4" style="color:#484f58">Brak aktywnych zapytań</td></tr>';

    // History table
    const hBody = document.querySelector('#history-table tbody');
    hBody.innerHTML = d.history.map(r => `<tr>
      <td>${r.start}</td>
      <td>${r.ip}</td>
      <td>${fmt(r.duration)}</td>
      <td class="status-${r.status}">${r.status === 'done' ? '✓' : '✗'}</td>
      <td class="question">${r.question || '—'}</td>
    </tr>`).join('');

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

    def do_GET(self):
        if self.path == "/api/stats":
            data = json.dumps(stats.snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            html = DASHBOARD_HTML.replace("PROXY_PORT", str(args.proxy_port))
            html = html.replace("LM_PORT", str(args.lm_port))
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
    lines.append(f"{BOLD}╔═══════════════════════════════════════════════════════════════╗{RESET}")
    lines.append(f"{BOLD}║  🖥️  LLM Proxy  :{args.proxy_port} → LM Studio :{args.lm_port}          {DIM}{now}{RESET}{BOLD}  ║{RESET}")
    lines.append(f"{BOLD}║  📊 Dashboard:  http://localhost:{args.dashboard_port}                      {BOLD}║{RESET}")
    lines.append(f"{BOLD}╠═══════════════════════════════════════════════════════════════╣{RESET}")

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

    # Aktywne zapytania
    lines.append(f"{BOLD}╠═══════════════════════════════════════════════════════════════╣{RESET}")
    if snap["active"]:
        lines.append(f"  {YELLOW}⏳ AKTYWNE ({len(snap['active'])}){RESET}")
        for r in snap["active"]:
            q = r["question"][:40] or "—"
            lines.append(f"    {r['ip']:<16s} {YELLOW}{format_time(r['duration']):>6s}{RESET}  {q}")
    else:
        lines.append(f"  {GREEN}✓ Brak aktywnych zapytań{RESET}")

    # Ostatnie zapytania
    lines.append(f"{BOLD}╠═══════════════════════════════════════════════════════════════╣{RESET}")
    lines.append(f"  {DIM}Ostatnie zapytania:{RESET}")
    for r in snap["history"][:8]:
        status = f"{GREEN}✓{RESET}" if r["status"] == "done" else f"{RED}✗{RESET}"
        q = r["question"][:35] or "—"
        lines.append(f"    {r['start']}  {r['ip']:<16s} {format_time(r['duration']):>6s}  {status}  {q}")
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


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Sprawdź czy LM Studio odpowiada
    print(f"🔍 Sprawdzam LM Studio na {LM_BASE}...")
    try:
        r = requests.get(f"{LM_BASE}/v1/models", timeout=3,
                         headers={"Authorization": f"Bearer {args.api_key}"} if args.api_key else {})
        if r.status_code == 200:
            models = [m["id"] for m in r.json().get("data", [])]
            print(f"   ✅ LM Studio online — modele: {', '.join(models)}")
        elif r.status_code == 401:
            print(f"   ⚠️  LM Studio wymaga auth — podaj --api-key")
            return
        else:
            print(f"   ⚠️  LM Studio odpowiada HTTP {r.status_code}")
    except requests.ConnectionError:
        print(f"   ❌ LM Studio nie odpowiada na {LM_BASE}")
        print(f"      Uruchom LM Studio i spróbuj ponownie.")
        return

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

    print(f"\n   Studenci łączą się na: http://<TWOJE_IP>:{args.proxy_port}")
    print(f"   Ctrl+C aby zakończyć\n")
    time.sleep(2)

    # Terminal dashboard loop
    try:
        terminal_loop()
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
