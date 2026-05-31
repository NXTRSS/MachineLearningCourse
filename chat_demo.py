#!/usr/bin/env python3
"""
Chat z narzędziami — standalone demo.

Uruchomienie:
    python chat_demo.py                              # auto-detect LLM
    python chat_demo.py --backend ollama              # wymuś Ollamę (pomiń LM Studio)
    python chat_demo.py --backend ollama -m e4b       # Ollama + konkretny model
    python chat_demo.py --port 4242                   # konkretny port LM Studio
    python chat_demo.py --port 4242 --api-key alk-2026  # LM Studio z kluczem API
    python chat_demo.py --server http://192.168.1.50:4242  # serwer prowadzącego (LAN)
    python chat_demo.py -s http://192.168.1.50:4242 -k alk-2026  # serwer + hasło
    python chat_demo.py --model gemma                 # partial match na nazwie modelu

Otwiera w przeglądarce interfejs ChatGPT-like z:
  - kalkulatorem, pogodą, bazą prezydentów, Wikipedią, wyszukiwarką DuckDuckGo
  - collapsible reasoning (🧠 tok myślenia)
  - agent loop (multi-step tool calling)
  - pamięcią konwersacji

Wymaga: LM Studio lub Ollama (lokalnie) albo serwer prowadzącego.
"""

import argparse
import json
import math
import urllib.parse
from datetime import datetime
from pathlib import Path

import requests
from pydantic import BaseModel, Field

from utils import connect_llm, ensure_package

ensure_package("ddgs")
ensure_package("wikipedia")
ensure_package("gradio")

# ── Argumenty CLI ─────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Chat z narzędziami — Function Calling demo")
parser.add_argument("--model", "-m", type=str, default=None,
                    help="Nazwa modelu (partial match, np. 'gemma')")
parser.add_argument("--port", "-p", type=int, default=None,
                    help="Port LM Studio na localhost (np. 4242)")
parser.add_argument("--server", "-s", type=str, default=None,
                    help="Pełny URL serwera prowadzącego (np. 'http://192.168.1.50:4242')")
parser.add_argument("--backend", "-b", type=str, default=None,
                    choices=["ollama", "lmstudio"],
                    help="Wymuszony backend: 'ollama' lub 'lmstudio' (domyślnie: auto-detect)")
parser.add_argument("--api-key", "-k", type=str, default=None,
                    help="API key do serwera LLM (np. LM Studio z auth)")
parser.add_argument("--reasoning", "-r", type=str, default="all",
                    choices=["all", "first", "none"],
                    help="Wyświetlanie toku myślenia: 'all' (każdy krok), 'first' (tylko pierwszy), 'none' (wyłączone)")
args = parser.parse_args()

# ── Połączenie z LLM ─────────────────────────────────────────────────

if args.port or args.server:
    # Bezpośrednie połączenie: --port (localhost) lub --server (pełny URL)
    from openai import OpenAI
    import instructor

    if args.server:
        server_url = args.server.rstrip("/")
        if not server_url.startswith("http"):
            server_url = f"http://{server_url}"
        base_url = f"{server_url}/v1" if not server_url.endswith("/v1") else server_url
    else:
        base_url = f"http://localhost:{args.port}/v1"

    key = args.api_key or "lm-studio"
    _needs_gui_auth = False

    def _try_connect(api_key):
        c = OpenAI(base_url=base_url, api_key=api_key)
        models = [m.id for m in c.models.list().data]
        return c, models

    try:
        client, models = _try_connect(key)
    except Exception as e:
        print(f"❌ Nie mogę połączyć się z {base_url}")
        print(f"   Szczegóły: {e}")
        raise SystemExit(1)

    # ── Sprawdź czy chat endpoint wymaga auth ──
    #    Celowo BEZ nagłówka Authorization — testujemy czy serwer
    #    odrzuci anonimowe zapytanie (401/403).
    if not args.api_key:
        import requests as _req
        try:
            probe = _req.post(
                f"{base_url}/chat/completions",
                json={"model": "test", "messages": []},
                timeout=3,
            )
            if probe.status_code in (401, 403):
                _needs_gui_auth = True
                print("🔑 Serwer wymaga hasła — podasz je w przeglądarce.")
        except Exception as e:
            err = str(e).lower()
            if "401" in err or "unauthorized" in err or "authentication" in err:
                _needs_gui_auth = True
                print("🔑 Serwer wymaga hasła — podasz je w przeglądarce.")

    MODEL_NAME = models[0] if models else "unknown"
    instructor_client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
else:
    LECTURER_SERVER = "http://ADRES_SERWERA:PORT"
    # Jeśli nie podano --api-key, zapytaj interaktywnie (ukryte znaki)
    api_key = args.api_key
    if api_key is None:
        import getpass
        api_key = getpass.getpass("🔑 Hasło serwera (Enter = brak): ") or None
    client, instructor_client, MODEL_NAME = connect_llm(
        lecturer_server=LECTURER_SERVER,
        api_key=api_key,
        backend=args.backend,
    )

if not client:
    print("❌ Nie znaleziono LLM-a. Uruchom LM Studio lub Ollamę.")
    raise SystemExit(1)

# ── Opcjonalny override modelu ──
if args.model:
    try:
        available = [m.id for m in client.models.list().data]
        match = next((m for m in available if args.model in m), None)
        if match:
            MODEL_NAME = match
        else:
            print(f"⚠️  Nie znaleziono modelu '{args.model}'")
            print(f"   Dostępne: {available}")
            print(f"   Używam: {MODEL_NAME}")
    except Exception:
        MODEL_NAME = args.model

print(f"✅ Model: {MODEL_NAME}")

# ── System prompt ─────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = (
    "Jesteś pomocnym asystentem. Odpowiadaj po polsku. "
    "ZAWSZE używaj dostępnych narzędzi gdy pytanie tego dotyczy — "
    "nie próbuj odpowiadać z pamięci."
)

# <|think|> włącza natywne myślenie Gemma-4 — inne modele go nie potrzebują
_is_gemma = "gemma" in MODEL_NAME.lower()
DEFAULT_SYSTEM_PROMPT = f"<|think|>{_BASE_SYSTEM_PROMPT}" if _is_gemma else _BASE_SYSTEM_PROMPT
if _is_gemma:
    print("💡 Gemma wykryta → dodaję <|think|> do system promptu")


# ═══════════════════════════════════════════════════════════════════════
#  NARZĘDZIA
# ═══════════════════════════════════════════════════════════════════════

def make_tool(name, description, args_model):
    """Tworzy definicję narzędzia FC z modelu Pydantic."""
    schema = args_model.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema},
    }


AVAILABLE_TOOLS = {}
tools_definition = []


# ── 1. Kalkulator ─────────────────────────────────────────────────────

def calculate(expression: str) -> str:
    """Wykonuje obliczenie matematyczne."""
    try:
        expression = expression.replace("^", "**")
        allowed = {
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "pi": math.pi, "abs": abs, "round": round, "pow": pow,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Błąd w obliczeniu '{expression}': {e}"


class CalculateArgs(BaseModel):
    expression: str = Field(..., description="Wyrażenie matematyczne, np. '2+2', 'sqrt(144)'")

AVAILABLE_TOOLS["calculate"] = calculate
tools_definition.append(
    make_tool("calculate", "Wykonuje obliczenie matematyczne. Użyj gdy trzeba coś policzyć.", CalculateArgs)
)


# ── 2. Pogoda ─────────────────────────────────────────────────────────

MOCK_WEATHER = {
    "Kraków":   {"temp": 18, "opis": "słonecznie",              "wilgotność": 45},
    "Warszawa": {"temp": 15, "opis": "pochmurno",               "wilgotność": 70},
    "Gdańsk":   {"temp": 12, "opis": "deszcz",                  "wilgotność": 85},
    "Wrocław":  {"temp": 20, "opis": "słonecznie",              "wilgotność": 40},
    "Poznań":   {"temp": 16, "opis": "częściowe zachmurzenie",  "wilgotność": 55},
}


def get_weather(city: str) -> str:
    """Sprawdza aktualną pogodę w podanym mieście."""
    try:
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        r = requests.get(url, timeout=5, headers={"Accept-Language": "pl"})
        r.raise_for_status()
        data = r.json()
        current = data["current_condition"][0]
        desc_list = current.get("lang_pl", current.get("weatherDesc", [{"value": "brak danych"}]))
        obs_time = current.get("localObsDateTime", "")
        time_info = f", dane z {obs_time}" if obs_time else ""
        return f"{city}: {current['temp_C']}°C, {desc_list[0]['value']}, wilgotność {current['humidity']}%{time_info} (wttr.in)"
    except Exception:
        pass
    if city in MOCK_WEATHER:
        m = MOCK_WEATHER[city]
        fake_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"{city}: {m['temp']}°C, {m['opis']}, wilgotność {m['wilgotność']}%, dane z {fake_time} (dane zastępcze)"
    return f"Brak danych pogodowych dla: {city}"


class WeatherArgs(BaseModel):
    city: str = Field(..., description="Nazwa miasta, np. 'Kraków'")

AVAILABLE_TOOLS["get_weather"] = get_weather
tools_definition.append(
    make_tool("get_weather", "Sprawdza aktualną pogodę w podanym mieście (temperatura, opis, wilgotność, godzina pomiaru).", WeatherArgs)
)


# ── 3. Baza prezydentów ──────────────────────────────────────────────

def load_presidents():
    md_path = Path("prezydenci_polski.md")
    if not md_path.exists():
        return []
    text = md_path.read_text(encoding="utf-8")
    presidents, current = [], {}
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("### ") and not line.startswith("####"):
            if current:
                presidents.append(current)
            current = {"imię": line[4:].strip()}
        elif line.startswith("- **") and ":**" in line:
            key = line.split("**")[1].rstrip(":")
            value = line.split(":** ", 1)[1] if ":** " in line else ""
            current[key.lower()] = value
    if current:
        presidents.append(current)
    return presidents


PREZYDENCI = load_presidents()


def search_presidents(query: str) -> str:
    """Przeszukuje bazę danych o prezydentach Polski (III RP)."""
    if not PREZYDENCI:
        return "Brak danych — nie znaleziono pliku prezydenci_polski.md"
    query_lower = query.lower()
    words = query_lower.split()
    scored = []
    for p in PREZYDENCI:
        all_text = " ".join(f"{k} {v}" for k, v in p.items()).lower()
        hits = sum(1 for w in words if w in all_text)
        if hits > 0:
            scored.append((hits / len(words), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    wyniki = []
    for score, p in scored:
        if score >= 0.5 or (not wyniki and score > 0):
            lines = [f"### {p.get('imię', '?')}"]
            for key, val in p.items():
                if key != "imię":
                    lines.append(f"  - {key}: {val}")
            wyniki.append("\n".join(lines))
    return "\n\n".join(wyniki) if wyniki else f"Nie znaleziono wyników dla: {query}"


class SearchPresidentsArgs(BaseModel):
    query: str = Field(..., description="Zapytanie, np. 'Kwaśniewski', 'najmłodszy prezydent'")

AVAILABLE_TOOLS["search_presidents"] = search_presidents
tools_definition.append(
    make_tool(
        "search_presidents",
        "Przeszukuje bazę prezydentów Polski (III RP). "
        "Pola: kadencja, wiek w dniu objęcia urzędu, partia, wykształcenie, kluczowe wydarzenia, mało znane fakty. "
        "Używaj KRÓTKICH zapytań — nazwisko lub słowo kluczowe.",
        SearchPresidentsArgs,
    )
)


# ── 4. Wikipedia ──────────────────────────────────────────────────────

import wikipedia
import wikipedia.wikipedia as _wiki_module
wikipedia.set_lang("pl")
_wiki_module.USER_AGENT = "ALK-MachineLearning-Course/1.0 (https://github.com/NXTRSS/MachineLearningCourse) python-wikipedia/1.4.0"


def search_wikipedia(query: str) -> str:
    """Przeszukuje Wikipedię i zwraca tytuł + streszczenie artykułu."""
    try:
        results = wikipedia.search(query, results=3)
        if not results:
            return f"Nie znaleziono artykułów dla: {query}"
        try:
            page = wikipedia.page(results[0])
        except wikipedia.DisambiguationError as e:
            page = wikipedia.page(e.options[0])
        return f"{page.title}\n\n{page.summary[:500]}\n\nŹródło: {page.url}"
    except Exception as e:
        return f"Błąd wyszukiwania Wikipedia: {e}"


class SearchWikipediaArgs(BaseModel):
    query: str = Field(..., description="Zapytanie do Wikipedii, np. 'fotosynteza', 'Wrocław'")

AVAILABLE_TOOLS["search_wikipedia"] = search_wikipedia
tools_definition.append(
    make_tool(
        "search_wikipedia",
        "Przeszukuje polską Wikipedię. Użyj do pytań encyklopedycznych (historia, nauka, geografia).",
        SearchWikipediaArgs,
    )
)


# ── 5. DuckDuckGo (web search) ───────────────────────────────────────

from ddgs import DDGS


def search_web(query: str) -> str:
    """Przeszukuje internet przez DuckDuckGo."""
    try:
        raw_results = DDGS().text(query, max_results=8)
        results = [r for r in raw_results if "wikipedia.org" not in r.get("href", "")][:3]
        if not results:
            return f"Brak wyników (poza Wikipedią) dla: {query}. Spróbuj search_wikipedia."
        parts = [f"Wyniki wyszukiwania: {query}\n"]
        for i, r in enumerate(results, 1):
            parts.append(f"{i}. {r['title']}\n   {r['body'][:200]}\n   {r['href']}")
        parts.append("\n💡 To tylko snippety. Użyj fetch_webpage(url) aby przeczytać pełną treść wybranej strony.")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Wyszukiwanie niedostępne (DuckDuckGo): {e}. Spróbuj Wikipedii."


class SearchWebArgs(BaseModel):
    query: str = Field(..., description="Zapytanie do wyszukiwarki")

AVAILABLE_TOOLS["search_web"] = search_web
tools_definition.append(
    make_tool(
        "search_web",
        "Przeszukuje internet przez DuckDuckGo. Zwraca tytuły i krótkie snippety. "
        "Sprawdź czy snippety zawierają konkretną odpowiedź na pytanie użytkownika — "
        "jeśli nie, użyj fetch_webpage na kilku URL-ach z wyników.",
        SearchWebArgs,
    )
)


# ── 5b. Fetch webpage ───────────────────────────────────────────────

from utils import fetch_webpage


class FetchWebpageArgs(BaseModel):
    url: str = Field(..., description="Pełny URL strony do pobrania, np. 'https://example.com/article'")

AVAILABLE_TOOLS["fetch_webpage"] = fetch_webpage
tools_definition.append(
    make_tool(
        "fetch_webpage",
        "Pobiera stronę internetową i zwraca czysty tekst. "
        "Użyj po search_web żeby przeczytać pełną treść strony z wyników wyszukiwania.",
        FetchWebpageArgs,
    )
)


# ── 6. get_population ────────────────────────────────────────────────

def get_population(city: str) -> str:
    """Zwraca przybliżoną liczbę mieszkańców miasta (dane orientacyjne)."""
    populations = {
        "Warszawa": 1_860_000, "Kraków": 804_000, "Wrocław": 674_000,
        "Łódź": 670_000, "Poznań": 546_000, "Gdańsk": 486_000,
        "Szczecin": 395_000, "Bydgoszcz": 331_000, "Lublin": 334_000,
        "Białystok": 298_000, "Katowice": 284_000, "Toruń": 196_000,
    }
    city_title = city.strip().title()
    if city_title in populations:
        pop = populations[city_title]
        return f"{city_title}: ~{pop:,} mieszkańców (dane orientacyjne)".replace(",", " ")
    return f"Brak danych o populacji dla: {city}. Spróbuj search_wikipedia."


class GetPopulationArgs(BaseModel):
    city: str = Field(..., description="Nazwa miasta, np. 'Kraków'")

AVAILABLE_TOOLS["get_population"] = get_population
tools_definition.append(
    make_tool(
        "get_population",
        "Zwraca przybliżoną liczbę mieszkańców polskiego miasta.",
        GetPopulationArgs,
    )
)


# ═══════════════════════════════════════════════════════════════════════
#  LAUNCH
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n🔧 Narzędzia: {list(AVAILABLE_TOOLS.keys())}")

    from chat_ui import launch_chat

    # Jeśli proxy wymaga auth → hasło w GUI tworzy nowego klienta
    _auth_callback = None
    if _needs_gui_auth:
        def _auth_callback(_username, password):
            """Gradio auth: zwraca nowego klienta (sukces) lub False (błąd)."""
            from openai import OpenAI
            try:
                new_client = OpenAI(base_url=base_url, api_key=password)
                new_client.models.list()  # test połączenia
                return new_client  # sukces → nowy klient
            except Exception:
                return False

    # Proxy / serwer zdalny → pytaj o imię w Gradio
    _is_proxy = args.port is not None or args.server is not None
    _ask_user = _is_proxy and not _needs_gui_auth

    launch_chat(
        client=client,
        model_name=MODEL_NAME,
        tools_definition=tools_definition,
        available_tools=AVAILABLE_TOOLS,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        reasoning=args.reasoning,
        auth_fn=_auth_callback,
        ask_username=_ask_user,
    )
