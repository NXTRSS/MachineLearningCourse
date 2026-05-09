#!/usr/bin/env python3
"""
Chat z narzędziami — standalone demo.

Uruchomienie:
    python chat_demo.py                              # auto-detect LLM
    python chat_demo.py --model gemma                 # konkretny model (partial match)
    python chat_demo.py --port 4242                   # konkretny port LM Studio
    python chat_demo.py --port 4242 --model gemma     # oba

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
                    help="Port LM Studio (np. 4242)")
args = parser.parse_args()

# ── Połączenie z LLM ─────────────────────────────────────────────────

if args.port:
    # Bezpośrednie połączenie na podany port
    from openai import OpenAI
    import instructor
    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="lm-studio")
    instructor_client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
    # Pobierz nazwę modelu z serwera
    try:
        models = [m.id for m in client.models.list().data]
        MODEL_NAME = models[0] if models else "unknown"
    except Exception:
        print(f"❌ Nie mogę połączyć się z localhost:{args.port}")
        raise SystemExit(1)
else:
    LECTURER_SERVER = "http://ADRES_SERWERA:PORT"
    client, instructor_client, MODEL_NAME = connect_llm(lecturer_server=LECTURER_SERVER)

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

DEFAULT_SYSTEM_PROMPT = (
    "Jesteś pomocnym asystentem. Odpowiadaj po polsku. "
    "ZAWSZE używaj dostępnych narzędzi gdy pytanie tego dotyczy — "
    "nie próbuj odpowiadać z pamięci."
)


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
        return f"{city}: {current['temp_C']}°C, {desc_list[0]['value']}, wilgotność {current['humidity']}% (źródło: wttr.in)"
    except Exception:
        pass
    if city in MOCK_WEATHER:
        m = MOCK_WEATHER[city]
        return f"{city}: {m['temp']}°C, {m['opis']}, wilgotność {m['wilgotność']}% (dane zastępcze)"
    return f"Brak danych pogodowych dla: {city}"


class WeatherArgs(BaseModel):
    city: str = Field(..., description="Nazwa miasta, np. 'Kraków'")

AVAILABLE_TOOLS["get_weather"] = get_weather
tools_definition.append(
    make_tool("get_weather", "Sprawdza aktualną pogodę w podanym mieście (temperatura, opis, wilgotność).", WeatherArgs)
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
    wyniki = []
    for p in PREZYDENCI:
        all_text = " ".join(f"{k} {v}" for k, v in p.items()).lower()
        if all(w in all_text for w in words):
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
        "Zawiera kadencje, partie, wykształcenie, kluczowe wydarzenia i mało znane fakty.",
        SearchPresidentsArgs,
    )
)


# ── 4. Wikipedia ──────────────────────────────────────────────────────

import wikipedia
wikipedia.set_lang("pl")


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
        return "\n\n".join(parts)
    except Exception as e:
        return f"Wyszukiwanie niedostępne (DuckDuckGo): {e}. Spróbuj Wikipedii."


class SearchWebArgs(BaseModel):
    query: str = Field(..., description="Zapytanie do wyszukiwarki")

AVAILABLE_TOOLS["search_web"] = search_web
tools_definition.append(
    make_tool(
        "search_web",
        "Przeszukuje internet przez DuckDuckGo. Użyj TYLKO gdy potrzebujesz aktualnych informacji, "
        "których nie ma na Wikipedii ani w bazie prezydentów.",
        SearchWebArgs,
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

    launch_chat(
        client=client,
        model_name=MODEL_NAME,
        tools_definition=tools_definition,
        available_tools=AVAILABLE_TOOLS,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
