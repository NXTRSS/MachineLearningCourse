#!/usr/bin/env python3
"""Buduje Function_Calling.ipynb programatycznie."""
import json
import uuid


def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
        "id": None,
    }


def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
        "id": None,
    }


def h6_collapsed(text):
    cell = md(text)
    cell["metadata"]["jp-MarkdownHeadingCollapsed"] = True
    return cell


def separator():
    return md("###### ")


cells = []

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 1: INTRO
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
# Function Calling — LLM jako "mózg" który wywołuje funkcje

## Jak deweloperzy budują AI asystentów i agenty?

### Plan warsztatu

1. **Idea**: LLM nie tylko generuje tekst — może *decydować* które funkcje wywołać
2. **Definiowanie narzędzi**: Opiszemy funkcje Pythona w formacie zrozumiałym dla LLM-a
3. **Pod maską**: Zobaczymy *dokładnie* co LLM widzi, jak wybiera funkcję, i co dostaje z powrotem
4. **Pydantic**: Nauczymy narzędzia zwracać *ustrukturyzowane* dane — i zobaczymy jak Pydantic automatycznie generuje opisy narzędzi
5. **Structured Output**: Zmusimy LLM-a do odpowiadania w *ścisłym formacie* — z uzasadnieniem i źródłami
6. **Pętla agentowa**: Zbudujemy mini-agenta który sam decyduje jakie narzędzia użyć

### Czym jest Function Calling?

Normalnie LLM dostaje pytanie i generuje tekst. Ale co jeśli pytanie wymaga:
- Sprawdzenia **aktualnej** pogody?
- Wykonania **obliczeń** matematycznych?
- Przeszukania **Wikipedii** albo **internetu**?

LLM nie ma do tego dostępu — ale może **poprosić** nasz program o wywołanie odpowiedniej funkcji.
To jest właśnie **Function Calling** (wywoływanie funkcji):

```
Użytkownik: "Jaka jest pogoda w Krakowie?"
    ↓
LLM myśli: "Potrzebuję funkcji get_weather z argumentem city='Kraków'"
    ↓
Nasz program: wywołuje get_weather("Kraków") → "18°C, słonecznie"
    ↓
LLM: "W Krakowie jest 18°C i słonecznie!"
```

**Tak działają ChatGPT, Claude, GitHub Copilot** — LLM + zestaw narzędzi + pętla decyzyjna."""))

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 2: SETUP
# ══════════════════════════════════════════════════════════════════════

cells.append(md("## 1. Konfiguracja środowiska"))

cells.append(code("""\
from utils import ensure_package

ensure_package("openai")
ensure_package("pydantic")
ensure_package("instructor")
ensure_package("wikipedia")
ensure_package("ddgs")

from openai import OpenAI
import requests
import json
import math
import random
import textwrap
from pathlib import Path

print("Pakiety załadowane!")\
"""))

# === LLM AUTO-DETECT ===
cells.append(md("""\
## 2. Połączenie z LLM-em

Poniższa komórka automatycznie szuka działającego LLM-a. Zanim ją uruchomisz:

<div style="background:#fff3cd; border-left:4px solid #ffc107; padding:12px; border-radius:4px;">

**Upewnij się, że LLM jest uruchomiony!**

- **LM Studio** — otwórz aplikację → załaduj model → upewnij się że serwer jest włączony (zakładka *Local Server*, zielony przycisk *Start Server*)
- **Ollama** — na macOS zwykle startuje automatycznie. Jeśli nie: otwórz terminal i wpisz `ollama serve`

Szczegóły instalacji: patrz `setup_local_llm.ipynb` lub `docs/LOKALNE_LLM.md`
</div>

**Function calling działa najlepiej z modelami:** `qwen3.5:4b+`, `gemma4:4b+`, `qwen3:8b+`"""))

cells.append(code("""\
from utils import connect_llm

client, MODEL_NAME, OLLAMA_URL = connect_llm()

if client:
    print(f"\\nKlient LLM gotowy!\\nModel: {MODEL_NAME}")\
"""))

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 3: NARZĘDZIA
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 3. Definiujemy nasze "narzędzia" (funkcje Pythona)

Zaczniemy od prostych funkcji, które LLM będzie mógł wywoływać.
Każda funkcja ma **docstring** — opis co robi. To jest kluczowe!
LLM czyta ten opis żeby zdecydować *kiedy* i *jak* użyć danej funkcji.

Zobaczmy co się stanie **pod maską**."""))

cells.append(code("""\
def get_weather(city: str) -> str:
    \"\"\"
    Sprawdza aktualną pogodę w podanym mieście.

    Args:
        city: Nazwa miasta, np. 'Kraków', 'Warszawa', 'Gdańsk'
    \"\"\"
    pogoda = {
        "Kraków": {"temp": 18, "opis": "słonecznie", "wilgotność": 45},
        "Warszawa": {"temp": 15, "opis": "pochmurno", "wilgotność": 70},
        "Gdańsk": {"temp": 12, "opis": "deszcz", "wilgotność": 85},
        "Wrocław": {"temp": 20, "opis": "słonecznie", "wilgotność": 40},
        "Poznań": {"temp": 16, "opis": "częściowe zachmurzenie", "wilgotność": 55},
    }
    if city in pogoda:
        p = pogoda[city]
        return f"Pogoda w {city}: {p['temp']}°C, {p['opis']}, wilgotność {p['wilgotność']}%"
    return f"Nie mam danych pogodowych dla miasta: {city}"


def calculate(expression: str) -> str:
    \"\"\"
    Wykonuje obliczenie matematyczne.

    Args:
        expression: Wyrażenie matematyczne, np. '2 + 2', 'sqrt(144)', '15 * 7'
    \"\"\"
    try:
        allowed = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                   "pi": math.pi, "abs": abs, "round": round, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Wynik: {expression} = {result}"
    except Exception as e:
        return f"Błąd w obliczeniu '{expression}': {e}"


def load_presidents():
    \"\"\"Ładuje dane o prezydentach z pliku prezydenci_polski.md\"\"\"
    md_path = Path("prezydenci_polski.md")
    if not md_path.exists():
        return []

    text = md_path.read_text(encoding="utf-8")
    presidents = []
    current = {}

    for line in text.split("\\n"):
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
    \"\"\"
    Przeszukuje bazę danych o prezydentach Polski (III RP).
    Znajduje informacje pasujące do zapytania.

    Args:
        query: Zapytanie, np. 'najmłodszy prezydent', 'katastrofa', 'Kwaśniewski'
    \"\"\"
    if not PREZYDENCI:
        return "Brak danych — nie znaleziono pliku prezydenci_polski.md"

    query_lower = query.lower()
    wyniki = []
    for p in PREZYDENCI:
        all_text = " ".join(str(v) for v in p.values()).lower()
        if query_lower in all_text:
            info = f"{p.get('imię', '?')} ({p.get('kadencja', '?')})"
            if 'kluczowe wydarzenia' in p:
                info += f": {p['kluczowe wydarzenia']}"
            wyniki.append(info)

    if wyniki:
        return "\\n".join(f"• {w}" for w in wyniki)
    return f"Nie znaleziono wyników dla zapytania: {query}"


AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_presidents": search_presidents,
}

print(f"Załadowano {len(PREZYDENCI)} prezydentów z pliku .md")
print(f"Zdefiniowano {len(AVAILABLE_TOOLS)} narzędzia:")
for name, func in AVAILABLE_TOOLS.items():
    print(f"  {name}(): {func.__doc__.strip().split(chr(10))[0]}")\
"""))

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 4: JSON SCHEMA
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 4. Opisanie narzędzi dla LLM-a (schemat JSON)

LLM nie widzi naszego kodu Pythona bezpośrednio. Musimy mu **opisać**
każde narzędzie w formacie JSON Schema — standardzie którego używają:
- OpenAI (ChatGPT)
- Anthropic (Claude)
- Ollama / LM Studio (lokalne modele)

Zobaczmy jak wygląda taki opis — to jest **dokładnie** to co LLM dostaje:"""))

cells.append(code("""\
tools_definition = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Sprawdza aktualną pogodę w podanym mieście. Użyj gdy użytkownik pyta o pogodę.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Nazwa miasta, np. 'Kraków', 'Warszawa'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Wykonuje obliczenie matematyczne. Użyj gdy trzeba coś policzyć.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Wyrażenie matematyczne, np. '2+2', 'sqrt(144)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_presidents",
            "description": "Przeszukuje bazę danych o prezydentach Polski. Użyj gdy pytanie dotyczy prezydentów RP.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Zapytanie do bazy, np. 'Kwaśniewski', 'katastrofa'"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

print("=== TO WIDZI LLM (definicje narzędzi) ===")
print(json.dumps(tools_definition, indent=2, ensure_ascii=False))\
"""))

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 5: PIERWSZY FC
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 5. Pierwszy function call — krok po kroku, pod maską!

Teraz wyślemy pytanie do LLM-a razem z opisem narzędzi.
LLM **sam zdecyduje** czy potrzebuje narzędzia i którego.

Zobaczmy **dokładnie** co się dzieje na każdym etapie."""))

cells.append(code("""\
user_question = "Jaka jest pogoda w Krakowie?"

if OLLAMA_URL:
    print("╔" + "═"*68 + "╗")
    print("║  KROK 1: Wysyłamy pytanie + opisy narzędzi do LLM-a            ║")
    print("╚" + "═"*68 + "╝")
    print(f"\\n  Pytanie użytkownika: \\"{user_question}\\"")
    print(f"  Dostępne narzędzia: {[t['function']['name'] for t in tools_definition]}")
    print(f"  → Wysyłam do {MODEL_NAME}...\\n")

    messages = [
        {"role": "system", "content": "Jesteś pomocnym asystentem. Odpowiadaj po polsku. Używaj narzędzi gdy to potrzebne."},
        {"role": "user", "content": user_question}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, tools=tools_definition, temperature=0.1
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        tc = msg.tool_calls[0]
        func_name = tc.function.name
        func_args = json.loads(tc.function.arguments)

        print("╔" + "═"*68 + "╗")
        print("║  KROK 2: LLM wybrał narzędzie!                                ║")
        print("╚" + "═"*68 + "╝")
        print(f"  Narzędzie: {func_name}")
        print(f"  Argumenty: {func_args}")

        result = AVAILABLE_TOOLS[func_name](**func_args)

        print(f"\\n╔" + "═"*68 + "╗")
        print("║  KROK 3: Wywołujemy funkcję i odsyłamy wynik do LLM-a         ║")
        print("╚" + "═"*68 + "╝")
        print(f"  Wynik funkcji: {result}")

        messages.append(msg)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        final = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.1)

        print(f"\\n╔" + "═"*68 + "╗")
        print("║  KROK 4: LLM formułuje ostateczną odpowiedź                   ║")
        print("╚" + "═"*68 + "╝")
        print(f"  {final.choices[0].message.content}")
    else:
        print(f"  LLM odpowiedział bez narzędzia: {msg.content[:200]}")
else:
    print("LLM niedostępny — uruchom LM Studio lub Ollamę.")\
"""))

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 6: LLM SAM WYBIERA
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 6. LLM sam wybiera narzędzie!

Zobaczmy jak LLM reaguje na **różne** pytania —
automatycznie wybiera odpowiednie narzędzie (lub żadne!)."""))

cells.append(code("""\
def ask_with_tools(question, verbose=True):
    if not OLLAMA_URL:
        print("LLM niedostępny.")
        return None

    messages = [
        {"role": "system", "content": "Jesteś pomocnym asystentem. Odpowiadaj po polsku. Używaj narzędzi gdy to potrzebne."},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, tools=tools_definition, temperature=0.1
    )

    assistant_msg = response.choices[0].message

    if not assistant_msg.tool_calls:
        if verbose:
            print(f"  Narzędzie: BRAK (LLM odpowiedział sam)")
            print(f"  Odpowiedź: {assistant_msg.content[:200]}")
        return assistant_msg.content

    tool_call = assistant_msg.tool_calls[0]
    func_name = tool_call.function.name
    func_args = json.loads(tool_call.function.arguments)

    if verbose:
        print(f"  Narzędzie: {func_name}({func_args})")

    result = AVAILABLE_TOOLS[func_name](**func_args)
    if verbose:
        print(f"  Wynik:     {result[:150]}")

    messages.append(assistant_msg)
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    final = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.1)
    final_answer = final.choices[0].message.content

    if verbose:
        print(f"  Odpowiedź: {final_answer[:200]}")
    return final_answer

print("Funkcja ask_with_tools() gotowa!")\
"""))

cells.append(code("""\
pytania = [
    "Jaka jest pogoda w Gdańsku?",
    "Ile to jest 17 razy 23?",
    "Kto był najmłodszym prezydentem Polski?",
    "Co to jest Python?",
    "Oblicz pierwiastek z 256 i powiedz mi pogodę w Poznaniu",
]

if OLLAMA_URL:
    for q in pytania:
        print(f"\\n{'═'*60}")
        print(f"PYTANIE: {q}")
        print(f"{'═'*60}")
        ask_with_tools(q)
else:
    print("LLM niedostępny.")\
"""))

# ══════════════════════════════════════════════════════════════════════
# ĆWICZENIE 1
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
### Ćwiczenie 1: Zmień description i zobacz co się stanie

`description` w definicji narzędzia jest **jedyną wskazówką** jaką LLM ma, żeby zdecydować
którego narzędzia użyć. Zobaczmy co się stanie, gdy go zepsujemy.

**Zadanie:**
1. Zmień `description` narzędzia `get_weather` na coś **mylącego** (np. *"Wykonuje obliczenia matematyczne"*)
2. Uruchom komórkę poniżej
3. Zadaj pytanie o pogodę — który tool wybierze LLM?
4. Przywróć oryginalny description"""))

cells.append(code("""\
# Ćwiczenie 1: Zmień description jednego narzędzia i zobacz efekt

# Zapisz oryginał
original_desc = tools_definition[0]["function"]["description"]

# --- TUTAJ ZMIEŃ description na coś mylącego ---

tools_definition[0]["function"]["description"] = ...  # Tutaj wpisz swój kod

# --- KONIEC ZMIANY ---

try:
    print("Testuję z ZMIENIONYM description:")
    print(f"  get_weather description: '{tools_definition[0]['function']['description']}'")
    if OLLAMA_URL:
        print()
        ask_with_tools("Jaka jest pogoda w Krakowie?")
except (TypeError, NameError):
    print('⬆️ Uzupełnij description powyżej')

# Przywróć oryginał!
tools_definition[0]["function"]["description"] = original_desc
print(f"\\nPrzywrócono oryginał: '{original_desc}'")\
"""))

cells.append(h6_collapsed(
    '###### <span style="color: #c17f24;">💡 Podpowiedź</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
Spróbuj np. `"Wykonuje obliczenia matematyczne"` — LLM powinien wybrać `calculate` zamiast `get_weather` gdy zapytasz o pogodę. Albo `"Szuka informacji o prezydentach"` — wtedy LLM wywoła `search_presidents`."""))
cells.append(h6_collapsed(
    '###### <span style="color: #5a8a6a;">✅ Rozwiązanie</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
```python
tools_definition[0]["function"]["description"] = "Wykonuje obliczenia matematyczne"
```

LLM dostanie pytanie *"Jaka jest pogoda w Krakowie?"* ale widzi, że `get_weather` "wykonuje obliczenia". Więc albo wybierze `calculate` (bo myśli że to jedyne narzędzie), albo odpowie sam bez narzędzia.

**Wniosek:** Jeden wyraz w description zmienia zachowanie modelu. Dlatego w produkcyjnych systemach opisy narzędzi to kluczowy element."""))
cells.append(separator())

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 7: DODAJ NARZĘDZIE
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 7. Dodaj własne narzędzie

Wiesz już jak LLM wybiera narzędzia na podstawie opisu.
Pora dodać **własne** narzędzie do zestawu!"""))

# ══════════════════════════════════════════════════════════════════════
# ĆWICZENIE 2
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
### Ćwiczenie 2: Dodaj własne narzędzie

Stwórz nową funkcję i dodaj jej opis do `tools_definition`.

**Zadanie:** Napisz narzędzie `get_population(city)` które zwraca przybliżoną liczbę
mieszkańców polskiego miasta. Potem przetestuj je pytaniem."""))

cells.append(code("""\
# Ćwiczenie 2: Dodaj narzędzie get_population

# Krok 1: Zdefiniuj funkcję

def get_population(city: str) -> str:

    pass  # Tutaj wpisz swój kod

# Krok 2: Dodaj do AVAILABLE_TOOLS

...  # Tutaj wpisz swój kod

# Krok 3: Dodaj definicję JSON Schema do tools_definition

...  # Tutaj wpisz swój kod

# --- TEST ---
try:
    print(f"Mamy {len(AVAILABLE_TOOLS)} narzędzi: {list(AVAILABLE_TOOLS.keys())}")
    print(f"\\nTest bezpośredni: {get_population('Kraków')}")
    if OLLAMA_URL:
        print()
        ask_with_tools("Ile mieszkańców ma Kraków?")
except (TypeError, NameError):
    print('⬆️ Uzupełnij kod powyżej')\
"""))

cells.append(h6_collapsed(
    '###### <span style="color: #c17f24;">💡 Podpowiedź</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
Wzoruj się na `get_weather` — słownik z danymi, sprawdzenie czy klucz istnieje, zwrócenie sformatowanego stringa. Do `tools_definition` dodaj element `.append({...})` z kluczami `type`, `function` (zawierającym `name`, `description`, `parameters`)."""))
cells.append(h6_collapsed(
    '###### <span style="color: #5a8a6a;">✅ Rozwiązanie</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
```python
def get_population(city: str) -> str:
    populacja = {
        "Warszawa": 1_860_000, "Kraków": 800_000, "Wrocław": 670_000,
        "Gdańsk": 470_000, "Poznań": 535_000, "Łódź": 660_000,
    }
    if city in populacja:
        return f"{city} ma około {populacja[city]:,} mieszkańców."
    return f"Nie mam danych o populacji dla: {city}"

AVAILABLE_TOOLS["get_population"] = get_population

tools_definition.append({
    "type": "function",
    "function": {
        "name": "get_population",
        "description": "Zwraca przybliżoną liczbę mieszkańców polskiego miasta.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Nazwa miasta, np. 'Kraków'"}
            },
            "required": ["city"]
        }
    }
})
```"""))
cells.append(separator())

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 8: PYDANTIC
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 8. Pydantic — ustrukturyzowane odpowiedzi narzędzi

Do tej pory nasze narzędzia zwracały **zwykłe stringi**: `"Pogoda w Krakowie: 18°C, słonecznie"`.
To działa, ale ma problem: LLM musi **parsować tekst** żeby wyciągnąć z niego dane.

Co gdyby narzędzia zwracały **ustrukturyzowane obiekty** z jasno nazwanymi polami?
Tu wchodzi **Pydantic** — biblioteka do definiowania schematów danych w Pythonie.

<div style="background:#e8f4f8; border-left:4px solid #2196F3; padding:12px; border-radius:4px;">
<b>Pydantic w jednym zdaniu:</b> Definiujesz klasę z polami i typami → Pydantic waliduje dane automatycznie.
To samo co JSON Schema, ale w Pythonie — i generuje JSON Schema za darmo!
</div>"""))

cells.append(code("""\
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# Zamiast zwracać string "Pogoda w Krakowie: 18°C, słonecznie, wilgotność 45%"
# definiujemy MODEL danych:

class WeatherReport(BaseModel):
    city: str = Field(..., description="Nazwa miasta")
    temperature_celsius: float = Field(..., description="Temperatura w stopniach Celsjusza")
    conditions: str = Field(..., description="Opis pogody, np. 'słonecznie', 'deszcz'")
    humidity_percent: int = Field(..., ge=0, le=100, description="Wilgotność w procentach")

class MathResult(BaseModel):
    expression: str = Field(..., description="Wyrażenie matematyczne wejściowe")
    result: float = Field(..., description="Wynik obliczenia")

# Zobaczmy jak wygląda instancja:
report = WeatherReport(city="Kraków", temperature_celsius=18.0, conditions="słonecznie", humidity_percent=45)
print(report)
print()
print("Dostęp do pól:")
print(f"  Miasto: {report.city}")
print(f"  Temperatura: {report.temperature_celsius}°C")
print(f"  Walidacja — wilgotność 150%?")
try:
    WeatherReport(city="X", temperature_celsius=0, conditions="x", humidity_percent=150)
except Exception as e:
    print(f"  Błąd! {e}")\
"""))

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 9: AUTO JSON SCHEMA
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 9. Automatyczne generowanie JSON Schema z Pydantic

Pisaliśmy JSON Schema ręcznie w sekcji 4. Było to żmudne.
Pydantic potrafi **wygenerować ten schemat automatycznie** z definicji klasy!

To jest potężne — zamiast pisać opisy narzędzi ręcznie, **definiujesz model Pydantic
i schemat generuje się sam**."""))

cells.append(code("""\
# Pydantic automatycznie generuje JSON Schema:
schema = WeatherReport.model_json_schema()

print("=== JSON Schema wygenerowany przez Pydantic ===")
print(json.dumps(schema, indent=2, ensure_ascii=False))

print("\\n\\n=== Porównaj z tym co pisaliśmy ręcznie (sekcja 4) ===")
print(json.dumps(tools_definition[0]["function"]["parameters"], indent=2, ensure_ascii=False))\
"""))

cells.append(md("""\
Widzisz? Pydantic wygenerował **dokładnie taki sam format** jak pisaliśmy ręcznie — z typami,
opisami pól, walidacją. A do tego dostajemy **walidację za darmo** (np. wilgotność 0-100).

Teraz użyjemy tego do zbudowania narzędzia Wikipedia — zdefiniujemy model Pydantic,
a JSON Schema wygeneruje się automatycznie."""))

# ══════════════════════════════════════════════════════════════════════
# ĆWICZENIE 3: WIKIPEDIA
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
### Ćwiczenie 3: Wikipedia jako narzędzie (z modelem Pydantic)

Pora na **prawdziwe** narzędzie sięgające do internetu!
Biblioteka `wikipedia` pozwala przeszukiwać Wikipedię programatycznie.

**Zadanie:**
1. Zdefiniuj model `WikiArticle` (Pydantic) z polami: `title`, `summary`, `url`
2. Napisz funkcję `search_wikipedia(query)` która zwraca dane w formacie JSON (z modelu Pydantic)
3. Dodaj definicję narzędzia do `tools_definition` (użyj `WikiArticle.model_json_schema()`)
4. Przetestuj pytaniem do LLM-a"""))

cells.append(code("""\
import wikipedia
wikipedia.set_lang("pl")

# Krok 1: Zdefiniuj model Pydantic

class WikiArticle(BaseModel):

    ...  # Tutaj wpisz swój kod — pola: title (str), summary (str, max 3 zdania), url (str)

# Krok 2: Napisz funkcję

def search_wikipedia(query: str) -> str:

    pass  # Tutaj wpisz swój kod

# Krok 3: Zarejestruj narzędzie

...  # Tutaj wpisz swój kod — AVAILABLE_TOOLS + tools_definition.append(...)

# --- TEST ---
try:
    print("Test bezpośredni:")
    print(search_wikipedia("Kraków"))
    print(f"\\nMamy {len(AVAILABLE_TOOLS)} narzędzi: {list(AVAILABLE_TOOLS.keys())}")
    if OLLAMA_URL:
        print("\\nTest przez LLM:")
        ask_with_tools("Co to jest fotosynteza?")
except (TypeError, NameError):
    print('⬆️ Uzupełnij kod powyżej')\
"""))

cells.append(h6_collapsed(
    '###### <span style="color: #c17f24;">💡 Podpowiedź</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
Model Pydantic: 3 pola z `Field(...)`. Funkcja: `wikipedia.search()` zwraca listę tytułów, `wikipedia.page()` zwraca stronę z polami `.title`, `.url`, `.summary`. Uważaj na `DisambiguationError` — złap go w `try/except`. Narzędzie zwraca `WikiArticle(...).model_dump_json()` — JSON z Pydantic modelu."""))
cells.append(h6_collapsed(
    '###### <span style="color: #5a8a6a;">✅ Rozwiązanie</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
```python
class WikiArticle(BaseModel):
    title: str = Field(..., description="Tytuł artykułu na Wikipedii")
    summary: str = Field(..., description="Streszczenie artykułu (pierwsze zdania)")
    url: str = Field(..., description="Link do artykułu")

def search_wikipedia(query: str) -> str:
    try:
        results = wikipedia.search(query, results=3)
        if not results:
            return f"Nie znaleziono artykułów dla: {query}"
        page = wikipedia.page(results[0])
        article = WikiArticle(
            title=page.title,
            summary=page.summary[:500],
            url=page.url
        )
        return article.model_dump_json(indent=2)
    except wikipedia.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0])
            article = WikiArticle(title=page.title, summary=page.summary[:500], url=page.url)
            return article.model_dump_json(indent=2)
        except Exception:
            return f"Znaleziono wiele wyników: {', '.join(e.options[:5])}"
    except Exception as e:
        return f"Błąd wyszukiwania: {e}"

AVAILABLE_TOOLS["search_wikipedia"] = search_wikipedia

wiki_schema = WikiArticle.model_json_schema()
tools_definition.append({
    "type": "function",
    "function": {
        "name": "search_wikipedia",
        "description": "Przeszukuje Wikipedię i zwraca streszczenie artykułu. Użyj gdy pytanie dotyczy wiedzy ogólnej, historii, nauki, geografii.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Zapytanie do Wikipedii, np. 'fotosynteza', 'Nikola Tesla'"}
            },
            "required": ["query"]
        }
    }
})
```"""))
cells.append(separator())

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 10: WEB SEARCH
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 10. Web search — DuckDuckGo

Wikipedia jest świetna dla encyklopedycznej wiedzy. Ale co z **aktualnymi wydarzeniami**?

Biblioteka `ddgs` pozwala przeszukiwać internet przez DuckDuckGo — bez klucza API!

<div style="background:#fff3cd; border-left:4px solid #ffc107; padding:12px; border-radius:4px;">
<b>Uwaga:</b> DuckDuckGo może czasem zwrócić błąd przy wielu zapytaniach (rate limiting).
Dlatego owijamy to w <code>try/except</code> — defensive coding w praktyce!
</div>"""))

cells.append(code("""\
from ddgs import DDGS

class SearchResult(BaseModel):
    title: str = Field(..., description="Tytuł wyniku wyszukiwania")
    snippet: str = Field(..., description="Fragment tekstu z wyniku")
    url: str = Field(..., description="Adres URL źródła")

class WebSearchResponse(BaseModel):
    query: str = Field(..., description="Wyszukiwane zapytanie")
    results: List[SearchResult] = Field(..., description="Lista znalezionych wyników")


def search_web(query: str) -> str:
    \"\"\"
    Przeszukuje internet przez DuckDuckGo.

    Args:
        query: Zapytanie, np. 'najnowsze wiadomości Polska'
    \"\"\"
    try:
        raw_results = DDGS().text(query, max_results=3)
        results = [
            SearchResult(title=r["title"], snippet=r["body"][:200], url=r["href"])
            for r in raw_results
        ]
        response = WebSearchResponse(query=query, results=results)
        return response.model_dump_json(indent=2)
    except Exception as e:
        return f"Wyszukiwanie niedostępne (DuckDuckGo): {e}. Spróbuj Wikipedii."


AVAILABLE_TOOLS["search_web"] = search_web
tools_definition.append({
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Przeszukuje internet przez DuckDuckGo. Użyj TYLKO gdy potrzebujesz aktualnych informacji, których nie ma na Wikipedii.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Zapytanie do wyszukiwarki"}
            },
            "required": ["query"]
        }
    }
})

print("Narzędzie search_web dodane!")
print(f"Mamy {len(AVAILABLE_TOOLS)} narzędzi: {list(AVAILABLE_TOOLS.keys())}")

print("\\nTest:")
result = search_web("Python programming language")
print(result[:300] + "...")\
"""))

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 11: INSTRUCTOR
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 11. Structured Output — LLM odpowiada strukturalnie

Do tej pory LLM zwracał **zwykły tekst** jako odpowiedź. Ale w produkcyjnych systemach
potrzebujemy **ustrukturyzowanej odpowiedzi** — z polami, źródłami, pewnością.

Biblioteka `instructor` pozwala **zmusić** LLM-a do odpowiadania w ścisłym formacie Pydantic.
Zamiast tekstu dostajemy **obiekt Pythona** z walidowanymi polami.

To jest ta sama technika, którą używają profesjonalne systemy AI:
- Zamiast *"Kraków ma 800 tysięcy mieszkańców"*
- Dostajemy `PopulationInfo(city="Kraków", population=800000, source="wikipedia")`"""))

cells.append(code("""\
import instructor

instructor_client = instructor.from_openai(
    OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="lm-studio" if "1234" in OLLAMA_URL else "ollama"),
    mode=instructor.Mode.JSON
) if OLLAMA_URL else None

class ToolReasoning(BaseModel):
    thinking: str = Field(..., description="Krótkie uzasadnienie — dlaczego wybrałeś to narzędzie (1-2 zdania)")
    needs_tool: bool = Field(..., description="Czy pytanie wymaga użycia narzędzia?")
    tool_name: Optional[str] = Field(None, description="Nazwa narzędzia do użycia (lub None jeśli nie potrzeba)")
    confidence: float = Field(..., ge=0, le=1, description="Pewność wyboru w skali 0-1")

if instructor_client:
    reasoning = instructor_client.chat.completions.create(
        model=MODEL_NAME,
        response_model=ToolReasoning,
        messages=[
            {"role": "system", "content": f"Masz dostępne narzędzia: {list(AVAILABLE_TOOLS.keys())}. Zdecyduj które narzędzie użyć."},
            {"role": "user", "content": "Jaka jest pogoda w Krakowie?"}
        ],
    )
    print("LLM zdecydował STRUKTURALNIE:")
    print(f"  Myślenie:   {reasoning.thinking}")
    print(f"  Potrzebuje: {reasoning.needs_tool}")
    print(f"  Narzędzie:  {reasoning.tool_name}")
    print(f"  Pewność:    {reasoning.confidence:.0%}")
else:
    print("LLM niedostępny.")\
"""))

cells.append(code("""\
# Przetestujmy na różnych pytaniach:

test_questions = [
    "Ile to jest 2 do potęgi 10?",
    "Kim był Lech Wałęsa?",
    "Co to jest algorytm sortowania?",
    "Jaka pogoda jest dzisiaj w Gdańsku?",
]

if instructor_client:
    for q in test_questions:
        r = instructor_client.chat.completions.create(
            model=MODEL_NAME,
            response_model=ToolReasoning,
            messages=[
                {"role": "system", "content": f"Masz narzędzia: {list(AVAILABLE_TOOLS.keys())}. Zdecyduj."},
                {"role": "user", "content": q}
            ],
        )
        tool_str = r.tool_name or "BRAK"
        print(f"  {q}")
        print(f"    → {tool_str} (pewność: {r.confidence:.0%}) — {r.thinking[:80]}")
        print()\
"""))

# ══════════════════════════════════════════════════════════════════════
# ĆWICZENIE 4: FACTCHECK
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
### Ćwiczenie 4: FactCheck — zweryfikuj twierdzenie strukturalnie

Połączmy function calling ze Structured Output!

**Zadanie:** Zdefiniuj model `FactCheck` i napisz funkcję, która:
1. Dostaje twierdzenie do sprawdzenia (np. *"Lech Wałęsa dostał Nagrodę Nobla w 1990"*)
2. Szuka dowodów na Wikipedii (narzędzie `search_wikipedia`)
3. Używa `instructor` żeby LLM zwrócił **ustrukturyzowany werdykt**: prawda / fałsz / nie da się zweryfikować

**Model FactCheck powinien mieć pola:**
- `claim` — sprawdzane twierdzenie
- `evidence` — znalezione dowody
- `verdict` — werdykt: `Literal["prawda", "fałsz", "nie da się zweryfikować"]`
- `confidence` — pewność (0-1)
- `source` — skąd pochodzi dowód"""))

cells.append(code("""\
# Ćwiczenie 4: FactCheck

# Krok 1: Zdefiniuj model

class FactCheck(BaseModel):

    pass  # Tutaj wpisz swój kod — 5 pól opisanych powyżej

# Krok 2: Napisz funkcję verify_claim

def verify_claim(claim: str) -> FactCheck:

    pass  # Tutaj wpisz swój kod

# --- TEST ---
try:
    twierdzenia = [
        "Lech Wałęsa dostał Pokojową Nagrodę Nobla w 1983 roku",
        "Kraków jest stolicą Polski",
        "Aleksander Kwaśniewski miał 41 lat gdy został prezydentem",
    ]
    for t in twierdzenia:
        fc = verify_claim(t)
        print(f"Twierdzenie: {t}")
        print(f"  Werdykt:  {fc.verdict} (pewność: {fc.confidence:.0%})")
        print(f"  Dowody:   {fc.evidence[:100]}")
        print(f"  Źródło:   {fc.source}")
        print()
except (TypeError, NameError, AttributeError):
    print('⬆️ Uzupełnij kod powyżej')\
"""))

cells.append(h6_collapsed(
    '###### <span style="color: #c17f24;">💡 Podpowiedź</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
Krok 1: Użyj `Literal["prawda", "fałsz", "nie da się zweryfikować"]` dla verdict, `Field(..., ge=0, le=1)` dla confidence. Krok 2: Najpierw wyszukaj na Wikipedii (`search_wikipedia(claim)`), potem podaj wynik jako kontekst do `instructor_client.chat.completions.create(response_model=FactCheck, ...)`."""))
cells.append(h6_collapsed(
    '###### <span style="color: #5a8a6a;">✅ Rozwiązanie</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
```python
class FactCheck(BaseModel):
    claim: str = Field(..., description="Sprawdzane twierdzenie")
    evidence: str = Field(..., description="Znalezione dowody potwierdzające lub obalające")
    verdict: Literal["prawda", "fałsz", "nie da się zweryfikować"] = Field(
        ..., description="Werdykt na podstawie dowodów"
    )
    confidence: float = Field(..., ge=0, le=1, description="Pewność werdyktu (0=zgaduję, 1=pewny)")
    source: str = Field(..., description="Skąd pochodzą dowody, np. URL Wikipedii")

def verify_claim(claim: str) -> FactCheck:
    wiki_data = search_wikipedia(claim)

    return instructor_client.chat.completions.create(
        model=MODEL_NAME,
        response_model=FactCheck,
        messages=[
            {"role": "system", "content":
             "Jesteś weryfikatorem faktów. Na podstawie DOWODÓW z Wikipedii oceń twierdzenie. "
             "Jeśli dowody nie wystarczają, powiedz 'nie da się zweryfikować'. Bądź uczciwy."},
            {"role": "user", "content":
             f"Twierdzenie: {claim}\\n\\nDowody z Wikipedii:\\n{wiki_data}"}
        ],
    )
```"""))
cells.append(separator())

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 12: PĘTLA AGENTOWA
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 12. Pętla agentowa — LLM który myśli i działa

Do tej pory obsługiwaliśmy **jedno** wywołanie narzędzia.
Ale co jeśli LLM potrzebuje **kilku** narzędzi po kolei?

Np. pytanie: *"Sprawdź na Wikipedii ile lat miał Kwaśniewski gdy został prezydentem i oblicz ile to dni"*
wymaga:
1. Wyszukania na Wikipedii (albo w bazie prezydentów)
2. Obliczenia matematycznego

To jest **pętla agentowa** — LLM wielokrotnie wywołuje narzędzia aż znajdzie odpowiedź.

```
while LLM chce wywołać narzędzie:
    1. LLM wybiera narzędzie i argumenty
    2. Nasz kod wywołuje funkcję
    3. Wynik wraca do LLM-a
    4. LLM decyduje: potrzebuję jeszcze? → wróć do 1.
                     mam wszystko? → generuję odpowiedź
```"""))

cells.append(code("""\
def agent(question, max_steps=5):
    if not OLLAMA_URL:
        print("LLM niedostępny.")
        return

    messages = [
        {"role": "system", "content":
         "Jesteś pomocnym asystentem. Odpowiadaj po polsku. "
         "Używaj narzędzi aby znaleźć potrzebne informacje. "
         "Możesz wywołać wiele narzędzi po kolei."},
        {"role": "user", "content": question}
    ]

    print(f"Pytanie: {question}")
    print(f"{'─'*60}")

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=tools_definition, temperature=0.1
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            print(f"\\n  Krok {step+1}: LLM generuje odpowiedź")
            print(f"  {'─'*50}")
            print(f"  {msg.content}")
            return msg.content

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"\\n  Krok {step+1}: LLM wywołuje → {func_name}({func_args})")

            if func_name in AVAILABLE_TOOLS:
                result = AVAILABLE_TOOLS[func_name](**func_args)
            else:
                result = f"Nieznane narzędzie: {func_name}"
            print(f"           Wynik ← \\"{result[:120]}...\\"" if len(result) > 120 else f"           Wynik ← \\"{result}\\"")

            messages.append(msg)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    print(f"\\n  (Osiągnięto limit {max_steps} kroków)")

print("Agent gotowy! Użyj: agent('twoje pytanie')")\
"""))

cells.append(code("""\
if OLLAMA_URL:
    print("Test 1: Pytanie o prezydentów + obliczenie")
    print("═"*60)
    agent("Ile lat miał Aleksander Kwaśniewski gdy skończył kadencję? Oblicz ile to dni.")

    print("\\n\\nTest 2: Wikipedia + pogoda")
    print("═"*60)
    agent("Co to jest fotosynteza i jaka jest pogoda w Krakowie?")\
"""))

# ══════════════════════════════════════════════════════════════════════
# ĆWICZENIE 5
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
### Ćwiczenie 5: Zadaj agentowi trudne pytanie

Wymyśl pytanie, które wymaga użycia **dwóch lub więcej** narzędzi.
Teraz masz do dyspozycji: pogodę, kalkulator, prezydentów, Wikipedię, i web search.

Obserwuj jak agent krok po kroku dochodzi do odpowiedzi."""))

cells.append(code("""\
# Ćwiczenie 5: Twoje pytanie do agenta

MOJE_PYTANIE = ...  # Tutaj wpisz swój kod — wymyśl pytanie łączące kilka narzędzi

try:
    if OLLAMA_URL:
        agent(MOJE_PYTANIE)
except (TypeError, NameError):
    print('⬆️ Wpisz swoje pytanie powyżej')\
"""))

cells.append(h6_collapsed(
    '###### <span style="color: #c17f24;">💡 Podpowiedź</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
Przykłady pytań łączących narzędzia:
- *"Sprawdź na Wikipedii kto wynalazł telefon i oblicz ile lat temu to było"*
- *"Kto zginął w katastrofie smoleńskiej i jaka jest pogoda w Krakowie?"*
- *"Znajdź na Wikipedii wysokość Mont Blanc i przelicz ją na stopy (1 stopa = 0.3048 m)"*"""))
cells.append(separator())

# ══════════════════════════════════════════════════════════════════════
# ĆWICZENIE 6 (BONUS)
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
### Ćwiczenie 6 (bonus): Zbuduj własnego asystenta

Wybierz **temat** i stwórz mini-asystenta z własnymi narzędziami:

| Temat | Pomysły na narzędzia |
|---|---|
| Asystent podróżniczy | pogoda + Wikipedia (miasta) + kalkulator (waluty) |
| Asystent filmowy | Wikipedia (filmy) + web search (recenzje) + kalkulator (oceny) |
| Asystent naukowy | Wikipedia + kalkulator + web search (artykuły) |

**Zadanie:**
1. Wybierz temat
2. Zdefiniuj system prompt dopasowany do tematu
3. Opcjonalnie: dodaj 1 nowe narzędzie specyficzne dla tematu
4. Przetestuj 3 pytaniami"""))

cells.append(code("""\
# Ćwiczenie 6 (bonus): Twój asystent

TEMAT = "..."  # np. "asystent podróżniczy"
SYSTEM_PROMPT = "..."  # Tutaj wpisz swój kod — opisz rolę asystenta

def my_agent(question):
    if not OLLAMA_URL:
        print("LLM niedostępny.")
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    print(f"[{TEMAT}] {question}")
    print(f"{'─'*60}")

    for step in range(5):
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=tools_definition, temperature=0.3
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            print(f"\\n  Odpowiedź: {msg.content}")
            return

        for tc in msg.tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  Krok {step+1}: {fn}({args})")
            result = AVAILABLE_TOOLS.get(fn, lambda **kw: "Nieznane narzędzie")(**args)
            messages.append(msg)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

try:
    if OLLAMA_URL and TEMAT != "...":
        my_agent("Twoje pierwsze pytanie testowe")
except (TypeError, NameError):
    print('⬆️ Uzupełnij TEMAT i SYSTEM_PROMPT powyżej')\
"""))

cells.append(h6_collapsed(
    '###### <span style="color: #c17f24;">💡 Podpowiedź</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
Dla asystenta podróżniczego: `SYSTEM_PROMPT = "Jesteś asystentem podróżniczym. Pomagasz planować wycieczki. Używaj Wikipedii do znalezienia informacji o miastach, pogody do sprawdzenia warunków, i kalkulatora do przeliczeń walutowych. Odpowiadaj po polsku, entuzjastycznie."`"""))
cells.append(separator())

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 13: PODSUMOWANIE
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 13. Podsumowanie

### Co zrobiliśmy?

1. **Zdefiniowaliśmy narzędzia** — zwykłe funkcje Pythona z opisami
2. **Opisaliśmy je dla LLM-a** — w formacie JSON Schema (standard branżowy)
3. **Zobaczyliśmy pod maską** — co dokładnie LLM widzi, jak wybiera funkcję
4. **Pydantic** — ustrukturyzowane odpowiedzi narzędzi + automatyczne JSON Schema
5. **Structured Output** — `instructor` wymusza format odpowiedzi LLM-a (z uzasadnieniem, pewnością, źródłami)
6. **Pętla agentowa** — LLM sam decyduje które narzędzia wywołać i w jakiej kolejności

### Tak działają prawdziwe produkty AI

| Produkt | Narzędzia | Pydantic / Structured Output |
|---|---|---|
| ChatGPT | Przeglądarka, DALL-E, Code Interpreter | Wewnętrznie (JSON mode) |
| Claude | Wyszukiwanie, analiza plików, MCP | Tool use + JSON Schema |
| GitHub Copilot | Czytanie/pisanie plików, terminal | Structured tool responses |
| Cursor / Windsurf | Edycja kodu, terminal, dokumentacja | Structured diffs |

Wszystkie działają na tej samej zasadzie: **LLM + zestaw narzędzi + pętla agentowa + strukturalne odpowiedzi**.

### Kluczowe wnioski

<div style="background:#e8f4f8; border-left:4px solid #2196F3; padding:12px; border-radius:4px;">

**1. Description decyduje o wszystkim** — LLM wybiera narzędzie na podstawie opisu, nie kodu (ćwiczenie 1)

**2. Pydantic to nie tylko walidacja** — automatycznie generuje JSON Schema, wymusza strukturę odpowiedzi

**3. `instructor` zamienia LLM w "wypełniacz formularzy"** — zamiast tekstu dostajesz obiekty z polami

**4. Defensive coding jest konieczny** — web search może nie zadziałać, Wikipedia może zwrócić disambiguation, LLM może wybrać złe narzędzie

</div>"""))


# ══════════════════════════════════════════════════════════════════════
# BUDOWANIE NOTEBOOKA
# ══════════════════════════════════════════════════════════════════════

nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (ml)",
            "language": "python",
            "name": "ml",
        },
        "language_info": {"name": "python", "version": "3.9.0"},
    },
    "cells": [],
}

for cell in cells:
    if cell["id"] is None:
        cell["id"] = str(uuid.uuid4())[:8]
    # Normalize source to list of strings with newlines
    if isinstance(cell["source"], str):
        lines = cell["source"].split("\n")
        cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    nb["cells"].append(cell)

with open("Function_Calling.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print(f"Zapisano Function_Calling.ipynb ({len(cells)} komórek)")
