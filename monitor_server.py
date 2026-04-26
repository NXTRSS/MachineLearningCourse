#!/usr/bin/env python3
"""
Monitor połączeń z serwerem LM Studio.
Użycie: python monitor_server.py
"""

import subprocess
import time
import requests
from datetime import datetime

LM_STUDIO_PORT = 1234
REFRESH_SECONDS = 3


def get_active_connections():
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{LM_STUDIO_PORT}"],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")[1:]  # pomiń nagłówek
        established = [l for l in lines if "ESTABLISHED" in l]
        listen = [l for l in lines if "LISTEN" in l]
        return established, len(listen) > 0
    except Exception:
        return [], False


def get_loaded_models():
    try:
        r = requests.get(f"http://localhost:{LM_STUDIO_PORT}/v1/models", timeout=2)
        if r.status_code == 200:
            return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        pass
    return []


def extract_ip(lsof_line):
    parts = lsof_line.split()
    for p in parts:
        if "->" in p:
            remote = p.split("->")[1]
            ip = remote.rsplit(":", 1)[0]
            return ip
    return "?"


def clear():
    print("\033[2J\033[H", end="")


def main():
    print("Monitor serwera LM Studio — Ctrl+C aby wyjść\n")
    time.sleep(1)

    while True:
        clear()
        now = datetime.now().strftime("%H:%M:%S")
        connections, server_up = get_active_connections()
        models = get_loaded_models()

        print(f"╔{'═'*50}╗")
        print(f"║  LM Studio Monitor  {now:>29}  ║")
        print(f"╠{'═'*50}╣")

        status = "✓ DZIAŁA" if server_up else "✗ WYŁĄCZONY"
        print(f"║  Serwer:   {status:<39}║")

        if models:
            print(f"║  Model:    {models[0]:<39}║")
        else:
            print(f"║  Model:    {'(brak załadowanego)':<39}║")

        print(f"╠{'═'*50}╣")
        print(f"║  Aktywne połączenia: {len(connections):<29}║")
        print(f"╠{'═'*50}╣")

        if connections:
            unique_ips = list(dict.fromkeys(extract_ip(c) for c in connections))
            for i, ip in enumerate(unique_ips, 1):
                print(f"║    {i}. {ip:<44}║")
        else:
            print(f"║    {'(brak połączeń)':<47}║")

        print(f"╚{'═'*50}╝")
        print(f"\n  Odświeżam co {REFRESH_SECONDS}s...")

        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nZatrzymano.")
