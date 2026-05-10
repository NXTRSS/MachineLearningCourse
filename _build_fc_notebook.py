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


def student_stub(source):
    """Komórka z zadaniem dla studenta — przed_zajeciami.py resetuje ją do szablonu."""
    cell = code(source)
    cell["metadata"]["tags"] = ["student-stub"]
    # Template = lista linii (format nbformat): każda z \n oprócz ostatniej
    lines = source.split("\n")
    cell["metadata"]["student_stub_template"] = [
        line + "\n" for line in lines[:-1]
    ] + [lines[-1]]
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

1. **Kalkulator + pierwszy Function Call**: Zbudujemy narzędzie i od razu zobaczymy je w akcji!
2. **Instructor**: Poznamy jak zmusić LLM-a do odpowiadania w ścisłym formacie Pydantic
3. **Pogoda i prezydenci**: Dodamy poważniejsze narzędzia — z prawdziwym API i bazą danych
4. **LLM wybiera**: Zobaczmy jak LLM sam decyduje którego narzędzia użyć
5. **Wikipedia i web search**: Podłączymy prawdziwe źródła wiedzy
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
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from IPython.display import display, Markdown
import requests
import json
import math
import urllib.parse
from pathlib import Path

print("Pakiety załadowane!")\
"""))

# === LLM AUTO-DETECT ===
cells.append(md("""\
## 2. Połączenie z LLM-em"""))

cells.append(md("""\
Poniższa komórka automatycznie szuka działającego LLM-a. Zanim ją uruchomisz:

<div style="background:#fff3cd; border-left:4px solid #ffc107; padding:12px; border-radius:4px;">

**Upewnij się, że LLM jest uruchomiony!**

- **LM Studio** (macOS / Windows / Linux) — otwórz aplikację → załaduj model → zakładka *Local Server* → zielony przycisk *Start Server*
- **Ollama** — otwórz terminal i wpisz `ollama serve`
  - *macOS:* jeśli zainstalowałeś przez `.dmg`, Ollama zwykle startuje automatycznie (ikonka w pasku menu)
  - *Windows:* Ollama działa jako usługa po instalacji — jeśli nie, uruchom ręcznie z menu Start
  - *Linux:* `ollama serve` w terminalu (lub `systemctl start ollama` jeśli zainstalowano przez `curl`)

Szczegóły instalacji: patrz `setup_local_llm.ipynb` lub `docs/LOKALNE_LLM.md`
</div>

**Function calling działa najlepiej z modelami:** `qwen3.5:4b+`, `gemma4:4b+`, `qwen3:8b+`"""))

cells.append(code("""\
from utils import connect_llm, extract_reasoning, print_reasoning, clean_content

# Jeśli nie masz lokalnego LLM-a, wpisz adres serwera prowadzącego (podany na zajęciach):
LECTURER_SERVER = "http://ADRES_SERWERA:PORT"  # ← prowadzący poda na zajęciach

client, instructor_client, MODEL_NAME = connect_llm(
    lecturer_server=LECTURER_SERVER,
    # model="gemma-4-e4b",  # ← odkomentuj, by wybrać konkretny model (np. "qwen", "llama")
)

# connect_llm zwraca DWA klienty:
#   client            → do function calling (LLM wybiera narzędzia)
#   instructor_client → do structured output (LLM odpowiada w formacie Pydantic)

if client:
    print(f"\\nKlient LLM gotowy!  Model: {MODEL_NAME}")
    print(f"Instructor:         {'tak' if instructor_client else 'nie'}")
    print()
    print("Mamy DWA klienty:")
    print("  client            → do function calling (LLM wybiera narzędzia)")
    print("  instructor_client → do structured output (LLM odpowiada w formacie Pydantic)")\
"""))

cells.append(code("""\
# ── System prompt — wspólny dla wszystkich funkcji ──────────────────
# Definiujemy go raz, w jednym miejscu. Dzięki temu:
#   1. Łatwo zmienić zachowanie WSZYSTKICH funkcji naraz
#   2. Studenci widzą, że system prompt to konfiguracja, nie magia
#   3. Unikamy copy-paste tego samego tekstu w 7 miejscach
#
# <|think|> na początku → włącza natywny tok myślenia w Gemma-4.
# Inne modele (Qwen3, Llama) go zignorują — nie przeszkadza.

DEFAULT_SYSTEM_PROMPT = (
    "<|think|>"
    "Jesteś pomocnym asystentem. Odpowiadaj po polsku. "
    "ZAWSZE używaj dostępnych narzędzi gdy pytanie tego dotyczy — "
    "nie próbuj odpowiadać z pamięci."
)

print(f"System prompt ({len(DEFAULT_SYSTEM_PROMPT)} znaków):")
print(f"  {DEFAULT_SYSTEM_PROMPT[:80]}...")\
"""))

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 3: KALKULATOR — PIERWSZE NARZĘDZIE + PIERWSZY FC
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 3. Kalkulator — twoje pierwsze narzędzie + Function Call!"""))

cells.append(md("""\
Zacznijmy od **najprostszego** możliwego narzędzia — kalkulatora.
Zobaczymy **cały cykl Function Calling** krok po kroku."""))

cells.append(code("""\
def calculate(expression: str) -> str:
    \"\"\"
    Wykonuje obliczenie matematyczne.

    Args:
        expression: Wyrażenie matematyczne, np. '2 + 2', 'sqrt(144)', '15 * 7'
    \"\"\"
    try:
        # LLM często pisze ^ zamiast ** (notacja matematyczna vs Python)
        expression = expression.replace("^", "**")
        allowed = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                   "pi": math.pi, "abs": abs, "round": round, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Błąd w obliczeniu '{expression}': {e}"


print("Test:")
print(calculate("sqrt(144) + 7 * 3"))
print(calculate("2 ** 10"))\
"""))

cells.append(md("""\
### Pierwszy Function Call — pod maską!"""))

cells.append(md("""\
Mamy narzędzie. Teraz **opiszemy** je dla LLM-a (w formacie JSON Schema)
i wyślemy pytanie. LLM **sam zdecyduje** czy potrzebuje kalkulatora.

Zobaczmy **dokładnie** co się dzieje na każdym etapie:"""))

cells.append(code("""\
# Opis narzędzia dla LLM-a (JSON Schema):
calc_tool = {
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
}

def calculate_function_call(user_prompt):
    \"\"\"Pełny cykl Function Calling: pytanie → LLM → narzędzie → odpowiedź.\"\"\"
    if not client:
        print("LLM niedostępny — uruchom LM Studio lub Ollamę.")
        return

    W = 60
    def box(text):
        print("╔" + "═"*W + "╗")
        print("║  " + text.ljust(W-4) + "  ║")
        print("╚" + "═"*W + "╝")

    box("KROK 1: Wysyłamy pytanie + opis narzędzia do LLM-a")
    print(f"\\n  Pytanie: \\"{user_prompt}\\"")
    print(f"  Narzędzie: calculate — {calc_tool['function']['description']}")
    print(f"  → Wysyłam do {MODEL_NAME}...\\n")

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, tools=[calc_tool], temperature=0.1
    )

    msg = response.choices[0].message

    print_reasoning(msg)

    content = clean_content(msg)
    if content and msg.tool_calls:
        print(f"  💬 LLM mówi: {content[:300]}")
        print()

    if msg.tool_calls:
        tc = msg.tool_calls[0]
        func_name = tc.function.name
        func_args = json.loads(tc.function.arguments)

        box("KROK 2: LLM wybrał narzędzie!")
        print(f"  Narzędzie: {func_name}")
        print(f"  Argumenty: {func_args}")

        result = calculate(**func_args)

        print()
        box("KROK 3: Wywołujemy funkcję i odsyłamy wynik do LLM-a")
        print(f"  Wynik: {result}")

        messages.append(msg)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        final = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.1)

        print()
        box("KROK 4: LLM formułuje ostateczną odpowiedź")
        print()
        display(Markdown(clean_content(final.choices[0].message)))
    else:
        print(f"  LLM odpowiedział bez narzędzia: {(content or '')[:200]}")

# --- Uruchomienie ---
calculate_function_call("Ile to jest 17 razy 23?")\
"""))

cells.append(md("""\
<div style="background:#d4edda; border-left:4px solid #28a745; padding:14px; border-radius:4px;">

**To jest cały cykl Function Calling!**

```
Użytkownik: "Ile to jest 17 razy 23?"
    ↓
LLM myśli: "Potrzebuję calculate(expression='17 * 23')"     ← LLM generuje argumenty
    ↓
Nasz kod: calculate("17 * 23") → {"result": 391.0}          ← Python liczy
    ↓
LLM: "17 razy 23 to 391"                                     ← LLM formułuje odpowiedź
```

Kluczowe: **LLM nie liczy sam** — prosi NASZ kod o obliczenie. My kontrolujemy narzędzia.

</div>

<div style="background:#e8f4f8; border-left:4px solid #2196F3; padding:12px; border-radius:4px; margin-top:10px;">

**🧠 Tok myślenia (reasoning):** Modele mogą pokazywać swój wewnętrzny tok myślenia na różne sposoby:
Qwen3 → osobne pole `reasoning_content`, Gemma-4 → tokeny `<|channel>thought` w odpowiedzi.
Nasza funkcja `extract_reasoning()` z `utils.py` wyciąga reasoning niezależnie od formatu.

</div>

Teraz dodamy więcej narzędzi — pogodę i bazę prezydentów.

Ale najpierw — **zauważyłeś, że `calc_tool` to ręcznie pisany JSON Schema?**
Był poręczny do nauki — widać dokładnie co LLM dostaje.
Ale pisanie tego ręcznie dla każdego narzędzia to błędogenna robota.

**Pydantic** umie wygenerować ten sam schemat automatycznie!

<div style="background:#e8f4f8; border-left:4px solid #2196F3; padding:12px; border-radius:4px;">
<b>Pydantic w jednym zdaniu:</b> Definiujesz klasę z polami i typami → Python automatycznie
waliduje dane i generuje schemat JSON — ten sam format którego używają OpenAI, Anthropic, Ollama.
</div>"""))

cells.append(code("""\
# Przypomnijmy — ręczny JSON Schema naszego kalkulatora:
print("RĘCZNIE napisany schemat (calc_tool → parameters):")
print(json.dumps(calc_tool["function"]["parameters"], indent=2, ensure_ascii=False))

# A teraz — Pydantic generuje to samo z klasy!
class CalculateArgs(BaseModel):
    expression: str = Field(..., description="Wyrażenie matematyczne, np. '2+2', 'sqrt(144)'")

print("\\nPYDANTIC wygenerowany schemat (CalculateArgs.model_json_schema()):")
print(json.dumps(CalculateArgs.model_json_schema(), indent=2, ensure_ascii=False))

# Jedyna różnica: Pydantic dodaje "title" — reszta identyczna!
# LLM ignoruje title, więc to nie ma znaczenia.\
"""))

cells.append(code("""\
# Helper — generuje definicję narzędzia z modelu Pydantic.
# Nigdy więcej ręcznego JSON Schema!

def make_tool(name, description, args_model):
    \"\"\"Tworzy definicję narzędzia FC z modelu Pydantic — zero ręcznego JSON Schema!\"\"\"
    schema = args_model.model_json_schema()
    schema.pop("title", None)  # OpenAI nie potrzebuje tytułu klasy
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        }
    }

# Test — porównajmy ręczny calc_tool z automatycznym:
calc_tool_auto = make_tool(
    "calculate",
    "Wykonuje obliczenie matematyczne. Użyj gdy trzeba coś policzyć.",
    CalculateArgs,
)

print("Ręczny calc_tool:")
print(json.dumps(calc_tool["function"]["parameters"], indent=2))
print("\\nAutomatyczny (make_tool + Pydantic):")
print(json.dumps(calc_tool_auto["function"]["parameters"], indent=2))
print("\\nOd teraz używamy make_tool() — Pydantic pisze JSON Schema za nas!")\
"""))

cells.append(md("""\
<div style="background:#fff3cd; border-left:4px solid #ffc107; padding:12px; border-radius:4px;">

**Podsumowanie: JSON Schema na dwa sposoby**

| | Ręcznie | Pydantic |
|---|---|---|
| **Jak** | Piszesz `{"type": "object", "properties": {...}}` | Piszesz `class Args(BaseModel): ...` |
| **Schema** | Ty go tworzysz | `model_json_schema()` generuje automatycznie |
| **Błędy** | Łatwo o literówkę | Pydantic waliduje za Ciebie |
| **Kiedy** | Żeby zrozumieć co siedzi pod spodem | W produkcji — zawsze |

Od tej pory będziemy definiować narzędzia przez **modele Pydantic + `make_tool()`**.

</div>"""))

# === 3b: INSTRUCTOR DEMO ===

cell_3b = md("""\
### 3b. *(opcjonalne)* Instructor — Structured Output""")
cell_3b["metadata"]["jp-MarkdownHeadingCollapsed"] = True
cells.append(cell_3b)

cells.append(md("""\
<div style="background:#e8f4f8; border-left:4px solid #2196F3; padding:12px; border-radius:4px;">
Ta sekcja pokazuje <b>drugi</b> sposób użycia Pydantic z LLM-em. Nie jest wymagana do dalszej pracy
z Function Calling — możesz ją pominąć i wrócić później.
</div>

<div style="background:#fff3cd; border-left:4px solid #ffc107; padding:12px; border-radius:4px; margin-top:8px;">
⏱️ <b>Uwaga:</b> Komórki w tej sekcji mogą wykonywać się <b>20–60 sekund</b> (zależnie od modelu).
Instructor wymusza format odpowiedzi przez prompt engineering — LLM musi wygenerować poprawny JSON,
a jeśli się pomyli, instructor powtarza zapytanie. To wolniejsze niż zwykły Function Calling.<br><br>
Jeśli korzystacie z <b>jednego serwera prowadzącego</b> — zapytania obsługiwane są jedno po drugim (kolejka).
Przy 20 osobach i ~30s na zapytanie, ostatnia osoba w kolejce może czekać nawet kilka minut.
Cierpliwości! 🙂
</div>

W Function Calling Pydantic opisuje **wejście** narzędzia (argumenty).
Ale jest też odwrotny kierunek: zmusić LLM-a żeby **odpowiedział** w formacie Pydantic.

Biblioteka **`instructor`** opakowuje klienta OpenAI i dodaje parametr `response_model=`.

```
Bez instructor:   LLM → "Wrocław to miasto na Dolnym Śląsku, ma ok. 670 tys. mieszkańców..."  (tekst)
Z instructor:     LLM → CityInfo(name="Wrocław", population=670000, region="Dolny Śląsk")   (obiekt!)
```

<div style="background:#e8f4f8; border-left:4px solid #2196F3; padding:12px; border-radius:4px;">
<b>Instructor pod maską:</b>
<ol>
<li>Bierze Twój model Pydantic → generuje JSON Schema</li>
<li>Wysyła schema + prompt do LLM-a: "Odpowiedz w tym formacie JSON"</li>
<li>Parsuje odpowiedź LLM-a przez Pydantic</li>
<li>Jeśli walidacja się nie powiedzie → wysyła LLM-owi feedback i prosi o poprawę (do 3 prób!)</li>
</ol>
</div>"""))

cells.append(code("""\
# Prosty model — informacja o mieście
class CityInfo(BaseModel):
    name: str = Field(..., description="Nazwa miasta")
    country: str = Field(..., description="Kraj")
    population_approx: int = Field(..., description="Przybliżona liczba mieszkańców")
    famous_for: str = Field(..., description="Z czego miasto jest znane (1 zdanie)")

# instructor w akcji — LLM MUSI odpowiedzieć jako CityInfo!
if instructor_client:
    city = instructor_client.chat.completions.create(
        model=MODEL_NAME,
        response_model=CityInfo,
        messages=[{"role": "user", "content": "Opowiedz o Wrocławiu"}],
    )
    print(f"Typ wyniku: {type(city).__name__}")
    print(f"\\nPola:")
    print(f"  city.name              = {city.name}")
    print(f"  city.country           = {city.country}")
    print(f"  city.population_approx = {city.population_approx}")
    print(f"  city.famous_for        = {city.famous_for}")
    print(f"\\nJSON:\\n{city.model_dump_json(indent=2)}")
else:
    print("instructor_client niedostępny — uruchom LLM-a i wróć do sekcji 2.")\
"""))

cells.append(md("""\
#### Co się stanie gdy pytanie nie pasuje do modelu?"""))

cells.append(md("""\
LLM **musi** wypełnić wszystkie pola — nawet gdy pytanie dotyczy czegoś zupełnie innego.
Zobaczmy co się stanie gdy zapytamy o smak lodów, a model wymaga danych o mieście:"""))

cells.append(code("""\
# MaybeCityInfo — model z polem "found" na wypadek pytań nie w temacie

class MaybeCityInfo(BaseModel):
    found: bool = Field(..., description="Czy pytanie dotyczy konkretnego miasta?")
    name: str = Field("", description="Nazwa miasta (puste jeśli found=False)")
    country: str = Field("", description="Kraj (puste jeśli found=False)")
    population_approx: int = Field(0, description="Przybliżona liczba mieszkańców (0 jeśli found=False)")
    famous_for: str = Field("", description="Z czego miasto jest znane (puste jeśli found=False)")
    error_message: str = Field("", description="Wyjaśnienie dlaczego nie znaleziono miasta (puste jeśli found=True)")

test_questions = [
    "Opowiedz o Gdańsku",
    "Opowiedz o Chrząszczyżewoszycach",
    "Jaki jest najlepszy smak lodów?",
    "Ile nóg ma pająk?",
]

if instructor_client:
    for q in test_questions:
        print(f"\\nPytanie: \\"{q}\\"")
        result = instructor_client.chat.completions.create(
            model=MODEL_NAME,
            response_model=MaybeCityInfo,
            messages=[{"role": "user", "content": q}],
        )
        if result.found:
            print(f"  ✓ Znaleziono: {result.name} ({result.country}), {result.population_approx:_} mieszkańców")
            print(f"    Znane z: {result.famous_for}")
        else:
            print(f"  ✗ Brak ustrukturyzowanej odpowiedzi — LLM: {result.error_message}")
else:
    print("instructor_client niedostępny.")\
"""))

cells.append(md("""\
#### Mechanizm retry — instructor sam naprawia błędy"""))

cells.append(md("""\
Jedną z największych zalet `instructor` jest **automatyczny retry z feedbackiem**.
Jeśli LLM zwróci niepoprawny JSON, instructor:
1. Łapie błąd parsowania/walidacji
2. **Odsyła go do LLM-a** jako wiadomość: *"Popraw odpowiedź, oto co było źle: ..."*
3. LLM poprawia się — i tak do `max_retries` prób

Zobaczmy to na żywo! Przechwytujemy odpowiedź LLM-a i **celowo psujemy JSON**
zanim instructor go zobaczy. Instructor wykryje błąd i poprosi LLM-a o poprawę:"""))

cells.append(code("""\
# --- Sabotaż! Przechwytujemy odpowiedź LLM-a i psujemy JSON ---
_orig_create = client.chat.completions.create
_call_count = [0]

def _sabotage(*args, **kwargs):
    _call_count[0] += 1

    # Przy retry — pokazujemy CO instructor wysyła do LLM-a
    if _call_count[0] == 2:
        msgs = kwargs.get('messages', [])
        print("  🔍 Co instructor wysyła do LLM-a przy retry:")
        print("  " + "─" * 55)
        for m in msgs:
            role = m.get('role', '?')
            content = str(m.get('content', ''))
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"  [{role}] {content}\\n")
        print("  " + "─" * 55 + "\\n")

    resp = _orig_create(*args, **kwargs)
    content = resp.choices[0].message.content

    if _call_count[0] == 1:  # tylko pierwszą odpowiedź psujemy
        print(f"  📡 Próba 1 — oryginalna odpowiedź LLM-a:")
        print(f"      {content[:200]}\\n")

        broken = content.replace("}", "###ZEPSUTE###")
        print(f"  💥 Psujemy JSON przed oddaniem instructorowi:")
        print(f"      {broken[:200]}\\n")
        resp.choices[0].message.content = broken
    else:
        print(f"  📡 Próba {_call_count[0]} — LLM poprawił odpowiedź:")
        print(f"      {content[:200]}\\n")
    return resp

client.chat.completions.create = _sabotage

# Tworzymy nowego instructor_client z "popsutym" klientem
import instructor
_sabotaged_ic = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)

class CityRetryDemo(BaseModel):
    name: str = Field(..., description="Nazwa miasta")
    country: str = Field(..., description="Kraj")
    population_approx: int = Field(..., description="Przybliżona liczba mieszkańców")

print("🔬 Demo: psujemy JSON + podglądamy retry!\\n")

city = _sabotaged_ic.chat.completions.create(
    model=MODEL_NAME,
    response_model=CityRetryDemo,
    messages=[{"role": "user", "content": "Opowiedz o Wrocławiu"}],
    max_retries=3,
)
print(f"  ✅ Sukces! {city.name}, {city.country}, {city.population_approx:,}")

# Przywracamy oryginalny client.create (ważne dla dalszych komórek!)
client.chat.completions.create = _orig_create\
"""))

cells.append(md("""\
<div style="background:#d4edda; border-left:4px solid #28a745; padding:12px; border-radius:4px;">
<b>Co się właśnie stało?</b>
<ol>
<li><b>[system]</b> — Instructor wstrzyknął JSON Schema do system message</li>
<li><b>[user]</b> — Nasze pytanie + instrukcja "Return JSON in codeblock"</li>
<li><b>[assistant]</b> — Zepsuta odpowiedź (instructor "widzi" ją jako oryginalną odpowiedź LLM-a)</li>
<li><b>[user]</b> — <code>"Correct your JSON ONLY RESPONSE, based on the following errors:"</code> + pełny traceback!</li>
<li>LLM czyta feedback i <b>sam się poprawia</b></li>
</ol>
W prawdziwym użyciu nie musisz nic psuć — instructor robi to <b>automatycznie</b>
gdy LLM zwróci niepoprawny format. To Twoja siatka bezpieczeństwa! 🛡️
</div>"""))

cells.append(md("""\
#### Function Calling vs. Structured Output — dwa różne mechanizmy!"""))

cells.append(md("""\
<div style="background:#fff3cd; border-left:4px solid #ffc107; padding:14px; border-radius:4px;">

| | **Function Calling** (`client`) | **Structured Output** (`instructor_client`) |
|---|---|---|
| **Co robi LLM** | **Wybiera** którą funkcję Pythona wywołać | **Wypełnia** schemat Pydantic danymi |
| **Kto wykonuje pracę** | Nasz kod Pythona (np. łączy się z API pogody) | Sam LLM (generuje dane "z głowy") |
| **Parametr** | `tools=tools_definition` | `response_model=MyModel` |
| **Wynik** | LLM woła funkcję → wynik → LLM formułuje tekst | Obiekt Pydantic (np. `CityInfo(...)`) |

**Analogia kuchenna:**
- **Function Calling** = LLM jest **kelnerem** — przyjmuje zamówienie i mówi kucharzowi (naszemu kodowi) co ugotować
- **Structured Output** = LLM jest **kucharzem** który musi podać danie dokładnie wg karty — nazwa, składniki, czas przygotowania, każde pole wypełnione, zero improwizacji

</div>"""))

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 4: POGODA — PRAWDZIWE API + FALLBACK
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 4. Pogoda — prawdziwe API + fallback"""))

cells.append(md("""\
Kalkulator pokazał nam cały cykl FC. Teraz zbudujemy **poważniejsze narzędzie**
— pobierające prawdziwą pogodę z API [wttr.in](https://wttr.in).

Jeśli API jest niedostępne — automatycznie przełącza się na dane zastępcze (mock).
Narzędzie zwraca **zwykły tekst** — LLM sobie z nim poradzi."""))

cells.append(code("""\
MOCK_WEATHER = {
    "Kraków":   {"temp": 18, "opis": "słonecznie",              "wilgotność": 45},
    "Warszawa": {"temp": 15, "opis": "pochmurno",               "wilgotność": 70},
    "Gdańsk":   {"temp": 12, "opis": "deszcz",                  "wilgotność": 85},
    "Wrocław":  {"temp": 20, "opis": "słonecznie",              "wilgotność": 40},
    "Poznań":   {"temp": 16, "opis": "częściowe zachmurzenie",  "wilgotność": 55},
}


def get_weather(city: str) -> str:
    \"\"\"
    Sprawdza aktualną pogodę w podanym mieście.
    Pobiera dane z wttr.in; jeśli API niedostępne — używa danych zastępczych.

    Args:
        city: Nazwa miasta, np. 'Kraków', 'Warszawa', 'Gdańsk'
    \"\"\"
    # Próba pobrania prawdziwych danych
    try:
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        r = requests.get(url, timeout=5, headers={"Accept-Language": "pl"})
        r.raise_for_status()
        data = r.json()
        current = data["current_condition"][0]
        desc_list = current.get("lang_pl", current.get("weatherDesc", [{"value": "brak danych"}]))
        temp = current["temp_C"]
        cond = desc_list[0]["value"]
        hum = current["humidity"]
        return f"{city}: {temp}°C, {cond}, wilgotność {hum}% (źródło: wttr.in)"
    except Exception:
        pass

    # Fallback — dane mockowane
    if city in MOCK_WEATHER:
        m = MOCK_WEATHER[city]
        return f"{city}: {m['temp']}°C, {m['opis']}, wilgotność {m['wilgotność']}% (dane zastępcze)"

    return f"Brak danych pogodowych dla: {city}"


# Test:
print("Test pogody:")
print(get_weather("Wrocław"))\
"""))

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 5: PREZYDENCI
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 5. Baza prezydentów Polski

Trzecie narzędzie — baza danych o prezydentach III RP wczytywana z pliku `prezydenci_polski.md`."""))

cells.append(code("""\
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
    Zawiera informacje o kadencjach, partiach, wykształceniu,
    kluczowych wydarzeniach i mało znanych faktach.

    Args:
        query: Zapytanie, np. 'najmłodszy prezydent', 'Kwaśniewski', 'mało znany fakt'
    \"\"\"
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
                if key != 'imię':
                    lines.append(f"  - {key}: {val}")
            wyniki.append("\\n".join(lines))

    if wyniki:
        return "\\n\\n".join(wyniki)
    return f"Nie znaleziono wyników dla zapytania: {query}"


AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_presidents": search_presidents,
}

print(f"Załadowano {len(PREZYDENCI)} prezydentów z pliku .md")
print(f"Mamy {len(AVAILABLE_TOOLS)} narzędzia: {list(AVAILABLE_TOOLS.keys())}")\
"""))


# ══════════════════════════════════════════════════════════════════════
# SEKCJA 6: JSON SCHEMA + WSZYSTKIE NARZĘDZIA + LLM WYBIERA
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 6. Trzy narzędzia — LLM wybiera!

Mamy kalkulator, pogodę i prezydentów. Teraz złóżmy je razem.

W sekcji 3 zdefiniowaliśmy `make_tool()` i `CalculateArgs` — teraz dodamy
modele Pydantic dla pozostałych narzędzi i sklejmy wszystko razem."""))

cells.append(code("""\
# Modele Pydantic dla argumentów każdego narzędzia
# (CalculateArgs już mamy z sekcji 3)

class GetWeatherArgs(BaseModel):
    city: str = Field(..., description="Nazwa miasta, np. 'Kraków', 'Warszawa'")

class SearchPresidentsArgs(BaseModel):
    query: str = Field(..., description="Pytanie o prezydentów, np. 'kto rządził najdłużej', 'mało znane fakty', 'Kwaśniewski'")

# Składamy tools_definition za pomocą make_tool()
tools_definition = [
    make_tool("get_weather",
              "Sprawdza aktualną pogodę w podanym mieście. Użyj gdy użytkownik pyta o pogodę.",
              GetWeatherArgs),
    make_tool("calculate",
              "Wykonuje obliczenie matematyczne. Użyj gdy trzeba coś policzyć.",
              CalculateArgs),
    make_tool("search_presidents",
              "Przeszukuje bazę danych o prezydentach Polski (III RP) — rozumie pytania semantycznie. "
              "Zawiera kadencje, partie, wykształcenie, kluczowe wydarzenia i mało znane fakty. "
              "Użyj gdy pytanie dotyczy prezydentów RP.",
              SearchPresidentsArgs),
]

print("LLM ma teraz 3 narzędzia do wyboru:")
for t in tools_definition:
    print(f"  {t['function']['name']}: {t['function']['description'][:60]}...")

print("\\nPrzykładowa definicja (get_weather):")
print(json.dumps(tools_definition[0], indent=2, ensure_ascii=False))\
"""))

cells.append(code("""\
# ── Model ToolReasoning ──────────────────────────────────────────────
# Modele mogą mieć wbudowany tok myślenia:
#   - Qwen3 → atrybut reasoning_content
#   - Gemma-4 → tokeny <|channel>thought w msg.content
# Jeśli model NIE ma natywnego reasoning — wymuszamy go przez instructor.
#
# ⚠️ UWAGA: Wymuszony reasoning = DWA OSOBNE zapytania — jedno o reasoning,
# drugie o właściwy tool call. Model może teoretycznie podjąć RÓŻNE decyzje!

class ToolReasoning(BaseModel):
    thinking: str = Field(..., description="Krótkie uzasadnienie — dlaczego wybrałeś to narzędzie (1-2 zdania)")
    needs_tool: bool = Field(..., description="Czy pytanie wymaga użycia narzędzia?")
    tool_name: Optional[str] = Field(None, description="Nazwa narzędzia do użycia (lub None)")
    confidence: float = Field(..., ge=0, le=1, description="Pewność wyboru w skali 0-1")


def ask_with_tools(question, verbose=True, show_reasoning=True):
    if not client:
        print("LLM niedostępny.")
        return None

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, tools=tools_definition, temperature=0.1
    )

    assistant_msg = response.choices[0].message

    # ── Reasoning: natywny (Qwen3, Gemma-4) → fallback na instructor ──
    if verbose and show_reasoning:
        native_reasoning = extract_reasoning(assistant_msg)
        if native_reasoning:
            print_reasoning(assistant_msg)
        elif instructor_client:
            try:
                forced = instructor_client.chat.completions.create(
                    model=MODEL_NAME,
                    response_model=ToolReasoning,
                    messages=[
                        {"role": "system", "content":
                         f"Masz dostępne narzędzia: {list(AVAILABLE_TOOLS.keys())}. "
                         "Zdecyduj które narzędzie użyć (lub żadne). "
                         "Odpowiedz PŁASKIM JSON-em z polami: thinking, needs_tool, tool_name, confidence."},
                        {"role": "user", "content": question}
                    ],
                )
                print(f"  🔍 Reasoning (wymuszony przez instructor):")
                print(f"     Myślenie:   {forced.thinking}")
                print(f"     Narzędzie:  {forced.tool_name or 'BRAK'} (pewność: {forced.confidence:.0%})")
                print()
            except Exception:
                pass

    content = clean_content(assistant_msg)
    if verbose and content and assistant_msg.tool_calls:
        print(f"  💬 LLM mówi: {content[:300]}")
        print()

    if not assistant_msg.tool_calls:
        if verbose:
            print(f"  Narzędzie: BRAK (LLM odpowiedział sam)")
            print()
            display(Markdown(content or ""))
        return content

    messages.append(assistant_msg)
    n_calls = len(assistant_msg.tool_calls)
    for i, tool_call in enumerate(assistant_msg.tool_calls, 1):
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        label = f"[{i}/{n_calls}] " if n_calls > 1 else ""
        if verbose:
            print(f"  {label}Narzędzie: {func_name}({func_args})")

        result = AVAILABLE_TOOLS.get(func_name, lambda **kw: "Nieznane narzędzie")(**func_args)
        if verbose:
            print(f"  {label}Wynik:     {result[:150]}")
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    final = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.1)
    final_answer = clean_content(final.choices[0].message)

    if verbose:
        print()
        display(Markdown(final_answer or ""))
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

if client:
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
### Ćwiczenie 1: Co ważniejsze — nazwa czy opis narzędzia?

LLM wybiera narzędzie na podstawie **nazwy** (`name`) i **opisu** (`description`).
Ale co ma większy wpływ na jego decyzję? Sprawdźmy!

**Część A — Zadanie: zmień opis i sprawdź efekt**

Zmień `description` narzędzia `get_weather` na coś kompletnie mylącego.
Zadaj pytanie o pogodę i obserwuj: czy LLM się pomylił?"""))

cells.append(student_stub("""\
# Ćwiczenie 1A: Zmień description get_weather na coś mylącego
import copy

# Kopia — nie modyfikujemy oryginału!
test_tools = copy.deepcopy(tools_definition)

# --- TUTAJ ZMIEŃ description ---

test_tools[0]["function"]["description"] = ...  # np. "Przeszukuje bazę prezydentów Polski"

# --- Funkcja do testowania (nie zmieniaj) ---
def test_1a(pytanie):
    new_desc = test_tools[0]["function"]["description"]
    if new_desc is ...:
        print("\\033[1;31m⬆️ Uzupełnij description powyżej!\\033[0m")
        return
    print(f"Nowy opis get_weather: '{new_desc}'")
    print(f"Ale nazwa to nadal:   '{test_tools[0]['function']['name']}'")
    print(f"Pytanie: '{pytanie}'")
    print()
    if client:
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": pytanie}
        ]
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, tools=test_tools, temperature=0.1
            )
            msg = response.choices[0].message
            print_reasoning(msg)
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                print(f"  LLM wybrał: {tc.function.name}({tc.function.arguments})")
            else:
                print(f"  LLM odpowiedział bez narzędzia: {(clean_content(msg) or '')[:200]}")
        except Exception as e:
            print(f"\\033[1;33m⚠️ Model zwrócił błąd: {e}\\033[0m")
            print("Spróbuj uruchomić komórkę ponownie — niektóre modele czasem się zawieszają.")

# ═══ TESTUJ TUTAJ — zmień pytanie i uruchom komórkę ponownie! ═══

test_1a("Jaka jest pogoda we Wrocławiu?")\
"""))

cells.append(h6_collapsed(
    '###### <span style="color: #c17f24;">Podpowiedź</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
Wpisz np.: `tools_definition[0]["function"]["description"] = "Przeszukuje bazę danych o prezydentach Polski"`"""))
cells.append(h6_collapsed(
    '###### <span style="color: #5a8a6a;">Rozwiązanie</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(code("""\
# Ćwiczenie 1A — rozwiązanie
import copy

test_tools = copy.deepcopy(tools_definition)
test_tools[0]["function"]["description"] = "Przeszukuje bazę danych o prezydentach Polski"

def test_1a(pytanie):
    print(f"Nowy opis get_weather: '{test_tools[0]['function']['description']}'")
    print(f"Ale nazwa to nadal:   '{test_tools[0]['function']['name']}'")
    print(f"Pytanie: '{pytanie}'")
    print()
    if client:
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": pytanie}
        ]
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, tools=test_tools, temperature=0.1
            )
            msg = response.choices[0].message
            print_reasoning(msg)
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                print(f"  LLM wybrał: {tc.function.name}({tc.function.arguments})")
            else:
                print(f"  LLM odpowiedział bez narzędzia: {(clean_content(msg) or '')[:200]}")
        except Exception as e:
            print(f"\\033[1;33m⚠️ Model zwrócił błąd: {e}\\033[0m")
            print("Spróbuj uruchomić komórkę ponownie — niektóre modele czasem się zawieszają.")

# ═══ TESTUJ TUTAJ ═══
test_1a("Jaka jest pogoda we Wrocławiu?")\
"""))
cells.append(separator())

cells.append(md("""\
<div style="background:#fff3cd; border-left:4px solid #ffc107; padding:14px; border-radius:4px;">

**Co zaobserwowałeś?**

Większość modeli **i tak wywoła `get_weather`** — bo nazwa `"get_weather"` jest zbyt silnym sygnałem!
LLM czyta **nazwę funkcji** i na jej podstawie podejmuje decyzję, często ignorując opis.

**Wniosek z części A:** Sama zmiana opisu może nie wystarczyć. Nazwa funkcji to bardzo silna wskazówka.

</div>"""))

cells.append(md("""\
### Ćwiczenie 1 — Część B: Cichy błąd (najgroźniejszy rodzaj!)

Skoro LLM opiera się na nazwie — ukryjmy ją! Użyjemy **neutralnych nazw** (`tool_alpha`, `tool_beta`)
i **zamienimy opisy** dwóch funkcji, które przyjmują **ten sam typ argumentu** (`city: str`).

Kluczowy trik: obie funkcje akceptują `"Kraków"` i obie zwracają sensowny wynik —
ale **w zupełnie innym kontekście**. Nie będzie żadnego błędu, żadnego crashu.
Będą po prostu **złe dane**."""))

cells.append(code("""\
# Ćwiczenie 1B (demo): Cichy błąd — dwie funkcje z tym samym parametrem + zamienione opisy

# Pomocnicza funkcja — informacje o mieście (populacja, zabytki)
def city_info(city: str) -> str:
    \"\"\"Zwraca informacje o polskim mieście — populacja, zabytki, historia.\"\"\"
    dane = {
        "Kraków": "Kraków: ~800 tys. mieszkańców, Wawel, Sukiennice, Uniwersytet Jagielloński, stolica Małopolski.",
        "Warszawa": "Warszawa: ~1.86 mln mieszkańców, stolica Polski, Pałac Kultury i Nauki, Stare Miasto.",
        "Gdańsk": "Gdańsk: ~470 tys. mieszkańców, Żuraw, Stocznia Gdańska, Westerplatte, Długi Targ.",
        "Poznań": "Poznań: ~535 tys. mieszkańców, Stary Rynek, Koziołki Poznańskie, Międzynarodowe Targi.",
        "Wrocław": "Wrocław: ~674 tys. mieszkańców, Ostrów Tumski, krasnale, Hala Stulecia.",
    }
    return dane.get(city, f"Brak informacji o mieście: {city}")

# Tworzymy DWA narzędzia z neutralnymi nazwami i TYM SAMYM parametrem (city)
class CityArg(BaseModel):
    city: str = Field(..., description="Nazwa miasta, np. 'Kraków'")

weather_desc = "Sprawdza aktualną pogodę w polskim mieście — temperaturę i warunki atmosferyczne."
city_desc    = "Zwraca informacje o polskim mieście — populacja, zabytki, historia."

# Tworzymy narzędzia z NEUTRALNYMI nazwami
tool_alpha = make_tool("tool_alpha", weather_desc, CityArg)
tool_beta  = make_tool("tool_beta",  city_desc,    CityArg)

# 🔀 SABOTAŻ: zamieniamy opisy!
tool_alpha["function"]["description"] = city_desc     # tool_alpha WYGLĄDA jak info o mieście
tool_beta["function"]["description"]  = weather_desc  # tool_beta  WYGLĄDA jak pogoda

# Ale mapowanie do PRAWDZIWYCH funkcji zostaje oryginalne:
sabotaged_dispatch = {
    "tool_alpha": get_weather,  # ma opis "info o mieście" ale NAPRAWDĘ sprawdza pogodę!
    "tool_beta":  city_info,    # ma opis "pogoda" ale NAPRAWDĘ zwraca info o mieście!
}

sabotaged_tools = [tool_alpha, tool_beta]

print("Sabotaż gotowy! Oto co LLM zobaczy:\\n")
for t in sabotaged_tools:
    n = t["function"]["name"]
    d = t["function"]["description"]
    real_fn = sabotaged_dispatch[n]
    print(f"  {n}:")
    print(f"    Opis:             \\"{d}\\"")
    print(f"    Prawdziwa f-ja:   {real_fn.__name__}()")
    print()\
"""))

cells.append(code("""\
# Ćwiczenie 1B (test): Wysyłamy pytanie — LLM widzi podmienione opisy

if client:
    # Nazwy narzędzi w sabotażu to tool_alpha/tool_beta — podajemy je do reasoning
    sabotaged_names = [t["function"]["name"] for t in sabotaged_tools]

    for question in ["Jaka jest pogoda we Wrocławiu?", "Opowiedz mi o Wrocławiu jako mieście"]:
        print(f"{'═'*60}")
        print(f"Pytanie: \\"{question}\\"\\n")

        # Wymuszony reasoning — zobaczmy CO model myśli o sabotażowych narzędziach
        if instructor_client:
            try:
                reasoning = instructor_client.chat.completions.create(
                    model=MODEL_NAME,
                    response_model=ToolReasoning,
                    messages=[
                        {"role": "system", "content":
                         f"Masz dostępne narzędzia: {sabotaged_names}. "
                         f"Opisy: {[t['function']['description'] for t in sabotaged_tools]}. Zdecyduj. "
                         "Odpowiedz PŁASKIM JSON-em z polami: thinking, needs_tool, tool_name, confidence."},
                        {"role": "user", "content": question}
                    ],
                )
                print(f"  🔍 Reasoning (wymuszony):")
                print(f"     Myślenie:   {reasoning.thinking}")
                print(f"     Narzędzie:  {reasoning.tool_name or 'BRAK'} (pewność: {reasoning.confidence:.0%})")
                print()
            except Exception:
                pass

        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=sabotaged_tools, temperature=0.1
        )
        msg = response.choices[0].message

        print_reasoning(msg)

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            chosen_name = tc.function.name
            chosen_args = json.loads(tc.function.arguments)
            real_func = sabotaged_dispatch[chosen_name]

            print(f"  LLM wybrał:        {chosen_name}({chosen_args})")
            print(f"  Prawdziwa funkcja:  {real_func.__name__}()")

            result = real_func(**chosen_args)
            print(f"  Wynik:             \\"{result}\\"")
            print()

            # Analiza
            if "pogoda" in question.lower():
                if real_func.__name__ == "get_weather":
                    print("  ✅ Poprawnie! Dostaliśmy pogodę.")
                else:
                    print("  🤦 CICHY BŁĄD! Pytaliśmy o pogodę, a dostaliśmy info o mieście.")
                    print("     Żadnego errora — funkcja się wykonała, wynik wygląda sensownie.")
                    print("     Ale to NIE jest pogoda!")
            else:
                if real_func.__name__ == "city_info":
                    print("  ✅ Poprawnie! Dostaliśmy info o mieście.")
                else:
                    print("  🤦 CICHY BŁĄD! Pytaliśmy o miasto, a dostaliśmy pogodę.")
                    print("     Żadnego errora — ale odpowiedź jest kompletnie nie na temat!")
        else:
            print(f"  LLM odpowiedział bez narzędzia: {(clean_content(msg) or '')[:200]}")
        print()
else:
    print("LLM niedostępny.")\
"""))

cells.append(md("""\
<div style="background:#f8d7da; border-left:4px solid #dc3545; padding:14px; border-radius:4px;">

**Cichy błąd jest GORSZY od crashu!**

Gdyby funkcja się wysypała — zobaczylibyśmy `Error` i wiedzielibyśmy, że coś nie gra.
Ale tutaj **wszystko wygląda OK**: funkcja się wykonała, wynik jest poprawny gramatycznie,
nie ma żadnego ostrzeżenia. Jedyny problem? **Dane są kompletnie nie te.**

W produkcyjnych systemach to prowadzi do:
- Chatbot podaje złe informacje z przekonaniem
- Agent wykonuje niewłaściwą akcję (np. kasuje plik zamiast go wyszukać)
- Użytkownik nie wie, że odpowiedź jest błędna

**Dlatego nazwy i opisy narzędzi to krytyczny element bezpieczeństwa AI systemów!**

</div>"""))

# ── FAŁSZYWE DANE W ŹRÓDLE ──

cells.append(md("""\
### Fałszywe dane w źródle — *"garbage in, garbage out"*

Cichy błąd może wynikać nie tylko z podmienionego opisu narzędzia.
Nawet gdy LLM wybierze **właściwą** funkcję — wynik jest tak dobry jak **dane źródłowe**.

W naszym pliku `prezydenci_polski.md` ukryliśmy dwa **całkowicie zmyślone** "mało znane fakty".
Sprawdźmy, czy LLM zwróci je z pełnym przekonaniem:"""))

cells.append(code("""\
# Zapytajmy o "mało znane fakty" — co zwróci nasz system?

for q in ["mało znany fakt", "kolekcjoner monet"]:
    print(f"Zapytanie: '{q}'")
    print(f"{search_presidents(q)}")
    print()\
"""))

cells.append(md("""\
<div style="background:#f8d7da; border-left:4px solid #dc3545; padding:14px; border-radius:4px;">

**Oba te "fakty" są całkowicie ZMYŚLONE!**

- Kwaśniewski **nie miał** żadnego popiersia Kleopatry od ambasadora Egiptu
- Duda **nie kolekcjonuje** antycznych monet z czasów Juliusza Cezara

A mimo to nasz system zwrócił je jako prawdziwe dane.
LLM nie ma jak sprawdzić czy źródło jest rzetelne — przekaże dalej wszystko co dostanie.

</div>

<div style="background:#e8f4f8; border-left:4px solid #2196F3; padding:12px; border-radius:4px; margin-top:10px;">

**Wnioski z Ćwiczenia 1:**

| Źródło błędu | Efekt | Wykrywalność |
|---|---|---|
| Zły opis narzędzia (ale znana nazwa) | LLM i tak trafia | Żaden — nam się udało! |
| Podmienione nazwy + opisy | **Cichy błąd** — złe narzędzie, dane nie na temat | Trudna — trzeba sprawdzić |
| Fałszywe dane w źródle | **Cichy błąd** — prawidłowe narzędzie, ale kłamstwo w danych | Bardzo trudna! |

Dlatego w produkcyjnych systemach AI kluczowe są:
**dobre opisy narzędzi** + **weryfikacja danych źródłowych** + **testy jakości odpowiedzi**

</div>"""))
cells.append(separator())

# ══════════════════════════════════════════════════════════════════════
# ĆWICZENIE 2
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
### Ćwiczenie 2: Dodaj własne narzędzie

Stwórz nową funkcję i dodaj jej opis do `tools_definition`.

**Zadanie:** Napisz narzędzie `get_population(city)` które zwraca przybliżoną liczbę
mieszkańców polskiego miasta.

**Dane do użycia** (GUS 2024):

| Miasto | Mieszkańcy | Ranking |
|---|---|---|
| Warszawa | ~1 860 000 | #1 |
| Kraków | ~800 000 | #2 |
| Wrocław | ~674 000 | #3 |
| Łódź | ~646 000 | #4 |
| Poznań | ~535 000 | #5 |
| Gdańsk | ~470 000 | #6 |

**Jak to zrobić:**
1. Przechowaj dane w **słowniku** `dict` — klucz to nazwa miasta, wartość to **tupla** `(populacja, ranking)`
2. Funkcja zwraca **f-string** z informacjami, np. `"Kraków: ~800,000 mieszkańców (#2 w Polsce)"`
3. Dodaj do `AVAILABLE_TOOLS` i do `tools_definition` (przez `make_tool`)"""))

cells.append(student_stub("""\
# Ćwiczenie 2: Dodaj narzędzie get_population

# Krok 1: Zdefiniuj funkcję

def get_population(city: str) -> str:

    pass  # Tutaj wpisz swój kod — zwróć f-string z informacjami o mieście

# Krok 2: Dodaj do AVAILABLE_TOOLS (nie zmieniaj)
AVAILABLE_TOOLS["get_population"] = get_population

# Krok 3: Zdefiniuj model argumentów i dodaj do tools_definition

class GetPopulationArgs(BaseModel):
    city: str = Field(..., description="Nazwa miasta, np. 'Kraków'")

tools_definition.append(
    make_tool("get_population",

              ...  # Tutaj wpisz swój kod — opis narzędzia (1 zdanie, po polsku)

              GetPopulationArgs)
)

# --- TEST (nie zmieniaj) ---
_test_result = get_population("Wrocław")
if _test_result is None:
    print("\\033[1;31m⬆️ Uzupełnij funkcję get_population! Teraz zwraca None (pass).\\033[0m")
elif "get_population" not in AVAILABLE_TOOLS:
    print("\\033[1;31m⬆️ Dodaj get_population do AVAILABLE_TOOLS!\\033[0m")
else:
    print(f"Mamy {len(AVAILABLE_TOOLS)} narzędzi: {list(AVAILABLE_TOOLS.keys())}")
    print(f"\\nTest bezpośredni: {get_population('Wrocław')}")
    if client:
        print()
        ask_with_tools("Ile mieszkańców ma Wrocław?")\
"""))

cells.append(h6_collapsed(
    '###### <span style="color: #c17f24;">Podpowiedź</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
Wzoruj się na `get_weather` — słownik z danymi, f-string dla wyniku. Do `tools_definition` użyj `make_tool()` — potrzebujesz model Pydantic dla **argumentów** (tu: `city: str`) + `tools_definition.append(make_tool("get_population", "...", GetPopulationArgs))`."""))
cells.append(h6_collapsed(
    '###### <span style="color: #5a8a6a;">Rozwiązanie</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(code("""\
# Ćwiczenie 2 — rozwiązanie

def get_population(city: str) -> str:
    dane = {
        "Warszawa": (1_860_000, 1), "Kraków": (800_000, 2), "Wrocław": (674_000, 3),
        "Gdańsk": (470_000, 6), "Poznań": (535_000, 5), "Łódź": (646_000, 4),
    }
    if city in dane:
        pop, rank = dane[city]
        return f"{city}: ~{pop:,} mieszkańców (#{rank} w Polsce)"
    return f"Brak danych o populacji dla: {city}"

AVAILABLE_TOOLS["get_population"] = get_population

class GetPopulationArgs(BaseModel):
    city: str = Field(..., description="Nazwa miasta, np. 'Kraków'")

tools_definition.append(
    make_tool("get_population", "Zwraca przybliżoną liczbę mieszkańców polskiego miasta.", GetPopulationArgs)
)

# --- TEST ---
print(f"Mamy {len(AVAILABLE_TOOLS)} narzędzi: {list(AVAILABLE_TOOLS.keys())}")
print(f"\\nTest bezpośredni: {get_population('Wrocław')}")
if client:
    print()
    ask_with_tools("Ile mieszkańców ma Wrocław?")\
"""))
cells.append(separator())

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 7: WIKIPEDIA
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 7. Wikipedia jako narzędzie

Pora na **prawdziwe** narzędzie sięgające do internetu!
Biblioteka `wikipedia` pozwala przeszukiwać Wikipedię programatycznie."""))

# ══════════════════════════════════════════════════════════════════════
# ĆWICZENIE 3: WIKIPEDIA
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
### Ćwiczenie 3: Zbuduj narzędzie Wikipedia

**Zadanie:**
1. Napisz funkcję `search_wikipedia(query)` która zwraca tytuł, streszczenie i URL artykułu jako tekst
2. Dodaj definicję narzędzia do `tools_definition` + do `AVAILABLE_TOOLS`
3. Przetestuj pytaniem do LLM-a"""))

cells.append(student_stub("""\
import wikipedia
wikipedia.set_lang("pl")

# Biblioteka wikipedia:
#   wikipedia.search("Wrocław")     → lista tytułów: ["Wrocław", "Wrocław (ujednoznacznienie)", ...]
#   wikipedia.page("Wrocław")       → obiekt strony z polami:
#       page.title   → "Wrocław"
#       page.summary → "Wrocław – miasto na prawach powiatu..."  (pełne streszczenie)
#       page.url     → "https://pl.wikipedia.org/wiki/Wrocław"
#
# Uwaga: wikipedia.page() może rzucić DisambiguationError — złap go w try/except.

def search_wikipedia(query: str) -> str:
    \"\"\"Przeszukuje Wikipedię i zwraca tytuł + streszczenie artykułu.\"\"\"
    try:
        results = wikipedia.search(query, results=3)
        if not results:
            return f"Nie znaleziono artykułów dla: {query}"

        try:
            page = wikipedia.page(results[0])
        except wikipedia.DisambiguationError as e:
            page = wikipedia.page(e.options[0])

        # ✏️ UZUPEŁNIJ: Zwróć f-stringa z informacjami ze strony.
        # Użyj page.title, page.summary (obetnij do 500 znaków: page.summary[:500])
        # i page.url. Połącz je w jednego f-stringa oddzielonego enterami (\\n\\n).
        # Wzór:  f"{tytuł}\\n\\n{streszczenie}\\n\\nŹródło: {url}"

        ...

    except Exception as e:
        return f"Błąd wyszukiwania Wikipedia: {e}"

# --- Rejestracja narzędzia ---
AVAILABLE_TOOLS["search_wikipedia"] = search_wikipedia

class SearchWikipediaArgs(BaseModel):

    query: str = Field(..., description=...  # ✏️ Tutaj wpisz opis argumentu — co to jest query?
    )

tools_definition.append(
    make_tool(
        "search_wikipedia",

        ...  # ✏️ Tutaj wpisz opis narzędzia (string) — kiedy LLM powinien go użyć?

        SearchWikipediaArgs,
    )
)

# --- TEST (nie zmieniaj) ---
_test_result = search_wikipedia("Wrocław")
if _test_result is None or _test_result is ...:
    print("\\033[1;31m⬆️ Uzupełnij return w search_wikipedia!\\033[0m")
elif "search_wikipedia" not in AVAILABLE_TOOLS:
    print("\\033[1;31m⬆️ Dodaj search_wikipedia do AVAILABLE_TOOLS!\\033[0m")
else:
    print("Test bezpośredni:")
    print(_test_result[:300])
    print(f"\\nMamy {len(AVAILABLE_TOOLS)} narzędzi: {list(AVAILABLE_TOOLS.keys())}")
    if client:
        print("\\nTest przez LLM:")
        ask_with_tools("Co to jest fotosynteza?")\
"""))

cells.append(h6_collapsed(
    '###### <span style="color: #c17f24;">Podpowiedź</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
Funkcja: `wikipedia.search()` zwraca listę tytułów, `wikipedia.page()` zwraca stronę z polami `.title`, `.url`, `.summary`. Uważaj na `DisambiguationError` — złap go w `try/except`. Narzędzie zwraca zwykły tekst (f-string z tytułem, streszczeniem i URL). Do rejestracji: model Pydantic dla **argumentów** + `make_tool()`."""))
cells.append(h6_collapsed(
    '###### <span style="color: #5a8a6a;">Rozwiązanie</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(code("""\
# Ćwiczenie 3 — rozwiązanie
import wikipedia
wikipedia.set_lang("pl")

def search_wikipedia(query: str) -> str:
    \"\"\"Przeszukuje Wikipedię i zwraca tytuł + streszczenie artykułu.\"\"\"
    try:
        results = wikipedia.search(query, results=3)
        if not results:
            return f"Nie znaleziono artykułów dla: {query}"

        try:
            page = wikipedia.page(results[0])
        except wikipedia.DisambiguationError as e:
            page = wikipedia.page(e.options[0])

        return f"{page.title}\\n\\n{page.summary[:500]}\\n\\nŹródło: {page.url}"

    except Exception as e:
        return f"Błąd wyszukiwania Wikipedia: {e}"

AVAILABLE_TOOLS["search_wikipedia"] = search_wikipedia

class SearchWikipediaArgs(BaseModel):
    query: str = Field(..., description="Zapytanie do Wikipedii, np. 'fotosynteza', 'Nikola Tesla'")

tools_definition.append(
    make_tool("search_wikipedia",
              "Przeszukuje Wikipedię i zwraca streszczenie artykułu. "
              "Użyj gdy pytanie dotyczy wiedzy ogólnej, historii, nauki, geografii.",
              SearchWikipediaArgs)
)

# --- TEST ---
print(f"Mamy {len(AVAILABLE_TOOLS)} narzędzi: {list(AVAILABLE_TOOLS.keys())}")
print(f"\\nTest: {search_wikipedia('Wrocław')[:200]}...")
if client:
    print()
    ask_with_tools("Kim był Mikołaj Kopernik?")\
"""))
cells.append(separator())

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 8: WEB SEARCH — DUCKDUCKGO
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 8. Web search — DuckDuckGo

Wikipedia jest świetna dla encyklopedycznej wiedzy. Ale co z **aktualnymi wydarzeniami**?

Biblioteka `ddgs` pozwala przeszukiwać internet przez DuckDuckGo — **bez klucza API**!

<div style="background:#e8f4f8; border-left:4px solid #2196F3; padding:12px; border-radius:4px;">
<b>Dlaczego DuckDuckGo a nie Google?</b> Google wymaga klucza API i ma płatne limity.
DuckDuckGo pozwala na darmowe wyszukiwanie programatyczne — idealne do prototypowania.
Uwaga: przy wielu zapytaniach może pojawić się rate limiting (tymczasowa blokada).
</div>"""))

cells.append(code("""\
from ddgs import DDGS


def search_web(query: str) -> str:
    \"\"\"
    Przeszukuje internet przez DuckDuckGo (pomija Wikipedię — od tego mamy osobne narzędzie).

    Args:
        query: Zapytanie, np. 'najnowsze wiadomości Polska'
    \"\"\"
    try:
        # Pobieramy więcej wyników, bo część odfiltrujemy (Wikipedia)
        raw_results = DDGS().text(query, max_results=8)
        # Filtrujemy Wikipedię — mamy osobny tool search_wikipedia
        results = [r for r in raw_results if "wikipedia.org" not in r.get("href", "")][:3]
        if not results:
            return f"Brak wyników (poza Wikipedią) dla: {query}. Spróbuj search_wikipedia."
        parts = [f"Wyniki wyszukiwania: {query}\\n"]
        for i, r in enumerate(results, 1):
            parts.append(f"{i}. {r['title']}\\n   {r['body'][:200]}\\n   {r['href']}")
        return "\\n\\n".join(parts)
    except Exception as e:
        return f"Wyszukiwanie niedostępne (DuckDuckGo): {e}. Spróbuj Wikipedii."


AVAILABLE_TOOLS["search_web"] = search_web

class SearchWebArgs(BaseModel):
    query: str = Field(..., description="Zapytanie do wyszukiwarki")

tools_definition.append(
    make_tool("search_web",
              "Przeszukuje internet przez DuckDuckGo. Użyj TYLKO gdy potrzebujesz aktualnych informacji, "
              "których nie ma na Wikipedii ani w bazie prezydentów.",
              SearchWebArgs)
)

print("Narzędzie search_web dodane!")
print(f"Mamy {len(AVAILABLE_TOOLS)} narzędzi: {list(AVAILABLE_TOOLS.keys())}")

print("\\nTest bezpośredni:")
result = search_web("Python programming language")
print(result[:300] + "..." if len(result) > 300 else result)\
"""))

cells.append(md("""\
Sprawdźmy, czy LLM wybierze `search_web` zamiast `search_wikipedia` — pytamy o coś **aktualnego i konkretnego**:"""))

cells.append(code("""\
if client:
    # Pytanie o aktualne wydarzenia → LLM powinien wybrać search_web (nie Wikipedię)
    ask_with_tools("Kto wygrał ostatni finał Ligi Mistrzów?")\
"""))

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 9: INSTRUCTOR — STRUCTURED OUTPUT
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 9. Combo: Function Calling + Structured Output

W łańcuchu function calli argumenty narzędzi są strukturalne (protokół `tools=`),
ale **końcowa odpowiedź** LLM-a to wolny tekst. Co jeśli chcemy ją ustrukturyzować?

<div style="background:#fff3cd; border-left:4px solid #ffc107; padding:12px; border-radius:4px;">

**Combo pattern (użyjemy go w ćwiczeniu 4):**
1. **Function Calling** (`client` + `tools=`) → zbieramy dane z narzędzi
2. **Structured Output** (`instructor_client` + `response_model=`) → formatujemy końcowy werdykt

Dzięki temu końcowa odpowiedź to nie tekst do parsowania, ale obiekt z polami: `verdict`, `confidence`, `source`.
</div>

Combo pattern w praktyce — funkcja `verify_claim()`:"""))

cells.append(code("""\
def verify_claim(claim: str) -> "FactCheck":
    \"\"\"
    Combo pattern:
      Krok 1 (FC): szukaj dowodów — najpierw search_presidents (LLM strukturyzuje query),
                    a jeśli nie znajdzie → FC z WSZYSTKIMI narzędziami (Wikipedia, web, itp.)
      Krok 2 (Structured Output): instructor formatuje werdykt w FactCheck
    \"\"\"

    # ── Krok 1a: FC → search_presidents (nasza baza z fikcyjnymi faktami!) ──
    pres_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content":
             "Znajdź informacje o tym twierdzeniu w bazie prezydentów. "
             "ZAWSZE użyj narzędzia search_presidents — nie odpowiadaj z pamięci."},
            {"role": "user", "content": claim}
        ],
        tools=[t for t in tools_definition if t["function"]["name"] == "search_presidents"],
        temperature=0.1,
    )
    pres_msg = pres_response.choices[0].message

    evidence = "Nie znaleziono"
    source = "brak"
    if pres_msg.tool_calls:
        tc = pres_msg.tool_calls[0]
        func_args = json.loads(tc.function.arguments)
        evidence = search_presidents(**func_args)
        source = "search_presidents"

    # ── Krok 1b: Jeśli baza prezydentów nie pomogła → FC ze wszystkimi narzędziami ──
    if "Nie znaleziono" in evidence:
        full_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content":
                 "Znajdź dowody na temat tego twierdzenia. Użyj dostępnych narzędzi. "
                 "ZAWSZE użyj narzędzia — nie odpowiadaj z pamięci."},
                {"role": "user", "content": claim}
            ],
            tools=tools_definition,
            temperature=0.1,
        )
        full_msg = full_response.choices[0].message
        if full_msg.tool_calls:
            tc = full_msg.tool_calls[0]
            func_name = tc.function.name
            func_args = json.loads(tc.function.arguments)
            evidence = AVAILABLE_TOOLS.get(func_name, lambda **kw: "Nieznane narzędzie")(**func_args)
            source = func_name

    # ── Krok 2: Structured Output — instructor formatuje werdykt ──
    # instructor wymusza format FactCheck. Hint o PŁASKIM JSONie to workaround
    # na mniejsze modele (np. Gemma) które owijają JSON w "properties".
    return instructor_client.chat.completions.create(
        model=MODEL_NAME,
        response_model=FactCheck,
        messages=[
            {"role": "system", "content":
             "Jesteś weryfikatorem faktów. Na podstawie DOWODÓW oceń twierdzenie. "
             "Jeśli dowody nie wystarczają, powiedz 'nie da się zweryfikować'. Bądź uczciwy. "
             "Odpowiedz PŁASKIM JSON-em — NIE owijaj w 'properties'."},
            {"role": "user", "content":
             f"Twierdzenie: {claim}\\n\\nDowody (źródło: {source}):\\n{evidence}"}
        ],
    )

print("Funkcja verify_claim() gotowa — potrzebuje modelu FactCheck (ćwiczenie 4).")\
"""))

cells.append(md("""\
### Ćwiczenie 4: FactCheck — zweryfikuj twierdzenie strukturalnie

Funkcja `verify_claim()` jest gotowa powyżej — brakuje jej tylko **modelu `FactCheck`**.

**Twoje zadanie:** Uzupełnij 5 pól Pydantic:
- `claim` — sprawdzane twierdzenie (str)
- `evidence` — znalezione dowody (str)
- `verdict` — werdykt: `Literal["prawda", "fałsz", "nie da się zweryfikować"]`
- `confidence` — pewność 0-1: `Field(..., ge=0, le=1)`
- `source` — skąd pochodzi dowód (str)

`instructor` wymusza, żeby LLM zwrócił obiekt dokładnie w formacie Twojego modelu.
Jeśli LLM pominie pole, instructor automatycznie ponawia zapytanie z feedbackiem!"""))

cells.append(student_stub("""\
# Ćwiczenie 4: Uzupełnij model FactCheck

class FactCheck(BaseModel):

    pass  # ✏️ Tutaj wpisz 5 pól opisanych powyżej

# --- TEST (nie zmieniaj) ---
if not instructor_client:
    print("instructor_client niedostępny — pomiń to ćwiczenie.")
elif not hasattr(FactCheck, 'model_fields') or 'verdict' not in FactCheck.model_fields:
    print("\\033[1;31m⬆️ Uzupełnij model FactCheck! Brakuje pola 'verdict'.\\033[0m")
else:
    twierdzenia = [
        "Lech Wałęsa dostał Pokojową Nagrodę Nobla w 1983 roku",
        "Aleksander Kwaśniewski posiadał popiersie Kleopatry",
        "Kraków jest stolicą Polski",
    ]
    for t in twierdzenia:
        try:
            fc = verify_claim(t)
            print(f"Twierdzenie: {t}")
            print(f"  Werdykt:  {fc.verdict} (pewność: {fc.confidence:.0%})")
            print(f"  Dowody:   {fc.evidence[:100]}")
            print(f"  Źródło:   {fc.source}")
        except Exception as e:
            print(f"Twierdzenie: {t}")
            print(f"  ⚠️ Błąd: {str(e)[:150]}")
            print("  (Mniejsze modele mogą nie generować poprawnego JSON-a — spróbuj z większym)")
        print()\
"""))

cells.append(h6_collapsed(
    '###### <span style="color: #c17f24;">Podpowiedź</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
Użyj `Literal["prawda", "fałsz", "nie da się zweryfikować"]` dla verdict, `Field(..., ge=0, le=1)` dla confidence. Każde pole potrzebuje `Field(..., description="...")`. Funkcja `verify_claim()` jest gotowa — wystarczy uzupełnić model."""))
cells.append(h6_collapsed(
    '###### <span style="color: #5a8a6a;">Rozwiązanie</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(code("""\
# Ćwiczenie 4 — rozwiązanie (tylko model — verify_claim jest gotowa powyżej)

class FactCheck(BaseModel):
    claim: str = Field(..., description="Sprawdzane twierdzenie")
    evidence: str = Field(..., description="Znalezione dowody potwierdzające lub obalające")
    verdict: Literal["prawda", "fałsz", "nie da się zweryfikować"] = Field(
        ..., description="Werdykt na podstawie dowodów"
    )
    confidence: float = Field(..., ge=0, le=1, description="Pewność werdyktu (0=zgaduję, 1=pewny)")
    source: str = Field(..., description="Skąd pochodzą dowody, np. 'baza prezydentów', 'Wikipedia'")

# --- TEST ---
if instructor_client:
    twierdzenia = [
        "Lech Wałęsa dostał Pokojową Nagrodę Nobla w 1983 roku",
        "Aleksander Kwaśniewski posiadał popiersie Kleopatry",
        "Kraków jest stolicą Polski",
    ]
    for t in twierdzenia:
        try:
            fc = verify_claim(t)
            print(f"Twierdzenie: {t}")
            print(f"  Werdykt:  {fc.verdict} (pewność: {fc.confidence:.0%})")
            print(f"  Dowody:   {fc.evidence[:100]}...")
            print(f"  Źródło:   {fc.source}")
        except Exception as e:
            print(f"Twierdzenie: {t}")
            print(f"  ⚠️ Model nie wygenerował poprawnego JSON-a: {str(e)[:150]}")
            print("  (Mniejsze modele czasem owijają JSON w 'properties' — spróbuj z większym modelem)")
        print()
else:
    print("instructor_client niedostępny — pomiń to ćwiczenie.")\
"""))
cells.append(separator())

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 10: PĘTLA AGENTOWA
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 10. Pętla agentowa — LLM który myśli i działa

Do tej pory obsługiwaliśmy **jedno** wywołanie narzędzia.
Ale co jeśli LLM potrzebuje **kilku** narzędzi po kolei?

Np. pytanie: *"Sprawdź w bazie prezydentów ile lat miał Kwaśniewski gdy został prezydentem i oblicz ile to dni"*
wymaga:
1. Wyszukania w bazie prezydentów
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
def agent(question, max_steps=6):
    if not client:
        print("LLM niedostępny.")
        return

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT + " Możesz wywołać wiele narzędzi po kolei."},
        {"role": "user", "content": question}
    ]

    print(f"Pytanie: {question}")
    print(f"{'─'*60}")

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=tools_definition, temperature=0.1
        )

        msg = response.choices[0].message

        print_reasoning(msg)

        content = clean_content(msg)
        if content and msg.tool_calls:
            print(f"  💬 LLM mówi: {content[:300]}")
            print()

        if not msg.tool_calls:
            print(f"\\n  Krok {step+1}: LLM generuje odpowiedź")
            print(f"  {'─'*50}")
            print()
            display(Markdown(content or ""))
            return content

        messages.append(msg)
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"\\n  Krok {step+1}: LLM wywołuje → {func_name}({func_args})")

            try:
                if func_name in AVAILABLE_TOOLS:
                    result = AVAILABLE_TOOLS[func_name](**func_args)
                else:
                    result = f"Nieznane narzędzie: {func_name}"
            except Exception as e:
                result = f"Błąd wywołania {func_name}: {e}"

            display_result = result[:120] + "..." if len(result) > 120 else result
            print(f"           Wynik ← \\"{display_result}\\"")

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    print(f"\\n  (Osiągnięto limit {max_steps} kroków)")

print("Agent gotowy! Użyj: agent('twoje pytanie')")\
"""))

cells.append(code("""\
if client:
    print("Test 1: Prezydenci + obliczenie")
    print("═"*60)
    agent("Ile lat miał Aleksander Kwaśniewski gdy skończył kadencję? Oblicz ile to w przybliżeniu dni.")

    print("\\n\\nTest 2: Wikipedia + pogoda")
    print("═"*60)
    agent("Co to jest fotosynteza i jaka jest pogoda w Krakowie?")\
"""))

# ══════════════════════════════════════════════════════════════════════
# TAJEMNICA PREZYDENCKA — DEMO
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
### Tajemnica prezydencka — skąd LLM to wie?

Sprawdźmy jak agent radzi sobie z pytaniami o **mało znane fakty** z naszej bazy.
Czy LLM użyje naszego lokalnego narzędzia, czy spróbuje szukać na Wikipedii?"""))

cells.append(code("""\
if client:
    print("Test: Mało znane fakty o prezydentach")
    print("═"*60)
    agent("Jakie mało znane fakty kryją polscy prezydenci? Sprawdź w bazie prezydentów, szukaj 'mało znany fakt'.")

    print("\\n\\nTest: Tajemnica gabinetu Kwaśniewskiego")
    print("═"*60)
    agent("Co ciekawego trzymał Aleksander Kwaśniewski w swoim gabinecie prezydenckim? Sprawdź w bazie prezydentów.")\
"""))

cells.append(md("""\
<div style="background:#fff3cd; border-left:4px solid #ffc107; padding:16px; border-radius:4px; margin:12px 0;">

**Uwaga dla studentów:** Fakty o "popiersiu Kleopatry" (Kwaśniewski) i "monetach Cezara" (Duda) są
**całkowicie fikcyjne** — dodaliśmy je do pliku `prezydenci_polski.md` specjalnie na potrzeby tego ćwiczenia.

Ten eksperyment pokazuje coś ważnego:
- LLM odpowiedział na pytanie o "tajemnicę" bo **nasze narzędzie** mu ją podało
- Tych informacji **nie ma** na Wikipedii ani w internecie — i nie da się ich zweryfikować
- LLM nie "wie" — on korzysta z **narzędzi** które mu dajemy

**To jest klucz do zrozumienia agentów AI: agent jest tak dobry (lub tak zły) jak jego narzędzia.**

Gdybyśmy podali LLM-owi fałszywe dane — odpowiedziałby fałszywie, z pełną pewnością.
Dlatego **jakość danych** i **wybór narzędzi** to najważniejsze decyzje przy budowaniu agentów.
</div>"""))

# ══════════════════════════════════════════════════════════════════════
# ĆWICZENIE 5
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
### Ćwiczenie 5: Zadaj agentowi trudne pytanie

Wymyśl pytanie, które wymaga użycia **dwóch lub więcej** narzędzi.
Teraz masz do dyspozycji: pogodę, kalkulator, prezydentów, Wikipedię, i web search.

Obserwuj jak agent krok po kroku dochodzi do odpowiedzi."""))

cells.append(student_stub("""\
# Ćwiczenie 5: Twoje pytanie do agenta

MOJE_PYTANIE = ...  # Tutaj wpisz swój kod — wymyśl pytanie łączące kilka narzędzi

if MOJE_PYTANIE is ...:
    print("\\033[1;31m⬆️ Wpisz swoje pytanie powyżej! Zamień ... na string.\\033[0m")
elif client:
    agent(MOJE_PYTANIE)\
"""))

cells.append(h6_collapsed(
    '###### <span style="color: #c17f24;">Podpowiedź</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
Przykłady pytań łączących narzędzia:
- *"Sprawdź na Wikipedii kto wynalazł telefon i oblicz ile lat temu to było"*
- *"Jaka jest pogoda w Krakowie i ile mieszkańców ma Kraków? Oblicz ile osób przypada na 1 stopień temperatury"*
- *"Znajdź na Wikipedii wysokość Mont Blanc i przelicz ją na stopy (1 stopa = 0.3048 m)"*
- *"Jakie hobby ma Andrzej Duda wg bazy prezydentów? Sprawdź na Wikipedii czy to prawda."*"""))
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

cells.append(student_stub("""\
# Ćwiczenie 6 (bonus): Twój asystent

TEMAT = "..."  # np. "asystent podróżniczy"
SYSTEM_PROMPT = "..."  # Tutaj wpisz swój kod — opisz rolę asystenta

def my_agent(question):
    if not client:
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

        messages.append(msg)
        for tc in msg.tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  Krok {step+1}: {fn}({args})")
            try:
                result = AVAILABLE_TOOLS.get(fn, lambda **kw: "Nieznane narzędzie")(**args)
            except Exception as e:
                result = f"Błąd: {e}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

if TEMAT == "..." or SYSTEM_PROMPT == "...":
    print("\\033[1;31m⬆️ Uzupełnij TEMAT i SYSTEM_PROMPT powyżej!\\033[0m")
elif client:
    my_agent("Twoje pierwsze pytanie testowe")\
"""))

cells.append(h6_collapsed(
    '###### <span style="color: #c17f24;">Podpowiedź</span> '
    '<span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>'
))
cells.append(md("""\
Dla asystenta podróżniczego: `SYSTEM_PROMPT = "Jesteś asystentem podróżniczym. Pomagasz planować wycieczki. Używaj Wikipedii do znalezienia informacji o miastach, pogody do sprawdzenia warunków, i kalkulatora do przeliczeń walutowych. Odpowiadaj po polsku, entuzjastycznie."`"""))
cells.append(separator())

# ══════════════════════════════════════════════════════════════════════
# SEKCJA BONUS: FC + SO PIPELINE
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## Bonus: FC + Structured Output w pipeline

Zobaczmy jeszcze jeden wzorzec — **łączenie Function Calling ze Structured Output w jednym narzędziu**.

Nasze narzędzia zwracają zwykły tekst — LLM sobie z nim radzi. Ale co jeśli narzędzie zwraca **surowy tekst**,
z którego chcemy wyciągnąć strukturę?

Np. Wikipedia zwraca akapit tekstu o Krakowie — a my chcemy obiekt `CityInfo(name=..., population=..., famous_for=...)`.

**Nasz kod Pythona** tego nie potrafi (parsowanie języka naturalnego). Ale **LLM potrafi**!
I tu `instructor` jest idealny — bo jeśli LLM nie wypełni wszystkich pól,
instructor złapie `ValidationError` i każe mu spróbować ponownie (do 3 prób).

```
┌─────────────────────────────────────────────────────────────┐
│  1. Function Calling     │  2. Structured Output            │
│  (client + tools=)       │  (instructor + response_model=)  │
│                          │                                   │
│  LLM: "wywołaj           │  LLM: "wyciągnij z tego tekstu   │
│   search_wikipedia(       │   pola: name, population,        │
│   query='Kraków')"        │   famous_for"                    │
│         ↓                │         ↓                         │
│  Nasz kod → Wikipedia    │  instructor → obiekt Pydantic     │
│         ↓                │  (z retry jeśli brakuje pól!)     │
│  Surowy tekst            │                                   │
└─────────────────────────────────────────────────────────────┘
```"""))

cells.append(code("""\
# Model: ustrukturyzowane info o mieście
class CityProfile(BaseModel):
    name: str = Field(..., description="Nazwa miasta")
    country: str = Field(..., description="Kraj")
    population_approx: Optional[int] = Field(None, description="Przybliżona liczba mieszkańców (jeśli podana)")
    famous_for: str = Field(..., description="Z czego miasto jest znane (1-2 zdania)")
    fun_fact: Optional[str] = Field(None, description="Ciekawostka (jeśli znaleziona w tekście)")


def smart_city_lookup(city_name: str) -> str:
    \"\"\"
    Pipeline: Wikipedia (surowy tekst) → LLM + instructor (strukturalny obiekt).

    Krok 1: Function Calling — pobieramy artykuł z Wikipedii
    Krok 2: Structured Output — LLM parsuje tekst → CityProfile
    \"\"\"
    # KROK 1: Pobierz surowy tekst z Wikipedii
    try:
        raw_text = search_wikipedia(city_name)
    except Exception:
        raw_text = f"Brak danych o mieście {city_name}"

    print(f"  [Pipeline krok 1] Wikipedia zwróciła {len(raw_text)} znaków surowego tekstu")
    print(f"  Fragment: \\"{raw_text[:120]}...\\"")

    # KROK 2: LLM parsuje tekst → obiekt Pydantic (z retry!)
    if not instructor_client:
        return raw_text  # fallback: zwróć surowy tekst

    try:
        profile = instructor_client.chat.completions.create(
            model=MODEL_NAME,
            response_model=CityProfile,
            messages=[
                {"role": "system", "content":
                 "Wyciągnij informacje o mieście z podanego tekstu. "
                 "Użyj TYLKO informacji z tekstu — nie wymyślaj."},
                {"role": "user", "content": f"Tekst źródłowy:\\n{raw_text}"}
            ],
        )
        print(f"  [Pipeline krok 2] Instructor → CityProfile (z retry jeśli trzeba)")
        return profile.model_dump_json(indent=2)
    except Exception as e:
        return f"Błąd parsowania: {e}\\n\\nSurowy tekst: {raw_text[:300]}"


# Test pipeline
if instructor_client:
    print("=== Pipeline: Wikipedia → LLM → CityProfile ===\\n")
    result = smart_city_lookup("Kraków")
    print(f"\\n  Wynik (ustrukturyzowany):\\n{result}")
else:
    print("instructor_client niedostępny — uruchom LLM-a i wróć do sekcji 2.")\
"""))

cells.append(code("""\
# Porównanie: surowy tekst z Wikipedii vs. ustrukturyzowany obiekt
if instructor_client:
    cities = ["Gdańsk", "Wrocław"]

    for city_name in cities:
        print(f"\\n{'═'*60}")
        print(f"MIASTO: {city_name}")
        print(f"{'═'*60}")
        result_json = smart_city_lookup(city_name)

        try:
            parsed = json.loads(result_json)
            print(f"\\n  Obiekt CityProfile:")
            for key, val in parsed.items():
                print(f"    {key}: {val}")
        except Exception:
            print(f"  (surowy wynik: {result_json[:200]})")
else:
    print("instructor_client niedostępny.")\
"""))

cells.append(md("""\
<div style="background:#e8f4f8; border-left:4px solid #2196F3; padding:14px; border-radius:4px;">

**Co się tu wydarzyło?**

1. **Function Calling** pobrał surowy tekst z Wikipedii (narzędzie `search_wikipedia`)
2. **Instructor** (Structured Output) wziął ten tekst i wyciągnął z niego ustrukturyzowane pola
3. Gdyby LLM pominął pole — instructor złapałby `ValidationError` i kazał spróbować ponownie

To jest wzorzec **ETL z LLM-em**: Extract (Wikipedia) → Transform (instructor) → Load (obiekt Pydantic).

**Kiedy to stosować?**
- Gdy źródło danych zwraca **nieustrukturyzowany tekst** (Wikipedia, web search, PDF, email...)
- A Ty potrzebujesz **konkretnych pól** do dalszego przetwarzania
- I chcesz **gwarancję formatu** (instructor z retry)

</div>"""))

# ══════════════════════════════════════════════════════════════════════
# SEKCJA 11: PODSUMOWANIE
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 11. Podsumowanie"""))

cells.append(md("""\
### Co zrobiliśmy?

1. **Pydantic + instructor** — ustrukturyzowane dane, JSON Schema, walidacja + LLM odpowiada jako obiekt Pythona
2. **Function Calling vs. Structured Output** — dwa różne mechanizmy, dwa klienty, komplementarne role
3. **Narzędzia** — pogoda (prawdziwe API!), kalkulator, baza prezydentów, Wikipedia, web search
4. **Pod maską** — co dokładnie LLM widzi, jak wybiera funkcję, JSON Schema ręcznie vs. automatycznie
5. **Combo pattern** — FC zbiera dane z narzędzi → Structured Output formatuje końcowy werdykt (FactCheck!)
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

**2. Pydantic to nie tylko walidacja** — automatycznie generuje JSON Schema, wymusza strukturę odpowiedzi narzędzi

**3. FC + Structured Output = combo** — FC zbiera dane z prawdziwych źródeł, instructor formatuje końcowy werdykt w obiekt z polami

**4. Agent jest tak dobry jak jego narzędzia** — fikcyjne fakty o prezydentach udowodniły, że LLM wierzy temu co mu podamy

**5. Defensive coding jest konieczny** — web search może nie zadziałać, Wikipedia może zwrócić disambiguation, pogoda może być niedostępna (fallback!)

</div>"""))


# ══════════════════════════════════════════════════════════════════════
# SEKCJA 12: BONUS — CHAT Z PAMIĘCIĄ
# ══════════════════════════════════════════════════════════════════════

cells.append(md("""\
## 12. Bonus: Chat z pamięcią konwersacji

`ask_with_tools()` traktuje każde pytanie osobno — nie pamięta poprzednich.
Co jeśli chcemy, żeby LLM pamiętał kontekst? Np.:

```
Ty:  "Jaka pogoda we Wrocławiu?"
LLM: "15°C, częściowe zachmurzenie"
Ty:  "A jaka populacja tego miasta?"       ← LLM musi pamiętać, że chodzi o Wrocław!
```

Rozwiązanie: **`messages` żyje poza funkcją** — każde pytanie widzi poprzednie."""))

cells.append(code("""\
def chat_with_tools(messages, question, verbose=True):
    \"\"\"ask_with_tools z pamięcią — messages jest listą modyfikowaną in-place.\"\"\"
    if not client:
        print("LLM niedostępny.")
        return None

    # Przy pierwszym pytaniu dodaj system prompt
    if not messages:
        messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, tools=tools_definition, temperature=0.1
    )
    assistant_msg = response.choices[0].message

    print_reasoning(assistant_msg)
    content = clean_content(assistant_msg)

    if not assistant_msg.tool_calls:
        if verbose:
            print(f"  Narzędzie: BRAK (LLM odpowiedział sam)\\n")
            display(Markdown(content or ""))
        messages.append({"role": "assistant", "content": content})
        return content

    messages.append(assistant_msg)
    n_calls = len(assistant_msg.tool_calls)
    for i, tool_call in enumerate(assistant_msg.tool_calls, 1):
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)
        label = f"[{i}/{n_calls}] " if n_calls > 1 else ""
        if verbose:
            print(f"  {label}Narzędzie: {func_name}({func_args})")
        result = AVAILABLE_TOOLS.get(func_name, lambda **kw: "Nieznane narzędzie")(**func_args)
        if verbose:
            print(f"  {label}Wynik:     {result[:150]}")
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    final = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, temperature=0.1
    )
    final_answer = clean_content(final.choices[0].message)
    messages.append({"role": "assistant", "content": final_answer})

    if verbose:
        print()
        display(Markdown(final_answer or ""))
    return final_answer

print("Funkcja chat_with_tools() gotowa!")\
"""))

cells.append(code("""\
# Testujemy! Każde kolejne pytanie pamięta kontekst poprzednich.

if client:
    history = []

    print("─── Pytanie 1 ───")
    chat_with_tools(history, "Jaka jest pogoda we Wrocławiu?")

    print("\\n─── Pytanie 2 (LLM pamięta 'Wrocław') ───")
    chat_with_tools(history, "A jaka populacja tego miasta?")

    print("\\n─── Pytanie 3 (kontynuacja) ───")
    chat_with_tools(history, "Znajdź coś o tym mieście na Wikipedii")

    print(f"\\n{'═'*60}")
    print(f"Historia konwersacji: {len(history)} wiadomości")\
"""))


# ══════════════════════════════════════════════════════════════════════
# SZKICE — zakomentowany kod do eksperymentów
# ══════════════════════════════════════════════════════════════════════
#
# --- chat_with_tools: wersja ask_with_tools z pamięcią konwersacji ---
#
# Idea: messages żyje poza funkcją → każde pytanie widzi poprzednie.
# Użycie:
#   history = []
#   chat_with_tools(history, "Jaka pogoda we Wrocławiu?")
#   chat_with_tools(history, "A jaka populacja tego miasta?")  # LLM pamięta "Wrocław"
#
# def chat_with_tools(messages, question, verbose=True):
#     """ask_with_tools z pamięcią — messages jest listą modyfikowaną in-place."""
#     if not client:
#         print("LLM niedostępny.")
#         return None
#
#     # Przy pierwszym pytaniu dodaj system prompt
#     if not messages:
#         messages.append({
#             "role": "system",
#             "content": "Jesteś pomocnym asystentem. Odpowiadaj po polsku. "
#                        "ZAWSZE używaj dostępnych narzędzi gdy pytanie tego dotyczy — "
#                        "nie próbuj odpowiadać z pamięci."
#         })
#
#     messages.append({"role": "user", "content": question})
#
#     response = client.chat.completions.create(
#         model=MODEL_NAME, messages=messages, tools=tools_definition, temperature=0.1
#     )
#     assistant_msg = response.choices[0].message
#
#     if not assistant_msg.tool_calls:
#         if verbose:
#             print(f"  Narzędzie: BRAK (LLM odpowiedział sam)\n")
#             display(Markdown(assistant_msg.content))
#         messages.append({"role": "assistant", "content": assistant_msg.content})
#         return assistant_msg.content
#
#     messages.append(assistant_msg)
#     n_calls = len(assistant_msg.tool_calls)
#     for i, tool_call in enumerate(assistant_msg.tool_calls, 1):
#         func_name = tool_call.function.name
#         func_args = json.loads(tool_call.function.arguments)
#         label = f"[{i}/{n_calls}] " if n_calls > 1 else ""
#         if verbose:
#             print(f"  {label}Narzędzie: {func_name}({func_args})")
#         result = AVAILABLE_TOOLS.get(func_name, lambda **kw: "Nieznane narzędzie")(**func_args)
#         if verbose:
#             print(f"  {label}Wynik:     {result[:150]}")
#         messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
#
#     final = client.chat.completions.create(
#         model=MODEL_NAME, messages=messages, temperature=0.1
#     )
#     final_answer = final.choices[0].message.content
#     messages.append({"role": "assistant", "content": final_answer})
#
#     if verbose:
#         print()
#         display(Markdown(final_answer))
#     return final_answer
#
#
# Przykład użycia w notebooku:
#
# history = []
# chat_with_tools(history, "Jaka jest pogoda we Wrocławiu?")
# chat_with_tools(history, "A jaka populacja tego miasta?")
# chat_with_tools(history, "Znajdź coś o tym mieście na Wikipedii")
# print(f"\nHistoria: {len(history)} wiadomości")


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
        "language_info": {"name": "python", "version": "3.11.0"},
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
