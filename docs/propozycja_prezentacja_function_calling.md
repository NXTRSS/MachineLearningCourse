# Prezentacja: Function Calling — LLM jako "mózg" który wywołuje funkcje

## Kontekst kursu

Studenci przyszli na te zajęcia z wiedzą z poprzednich spotkań:
- ML Wprowadzenie (podstawy)
- Regresja liniowa
- Regresja logistyczna
- Sieci neuronowe (FFNN, CNN)
- Reprezentacja tekstu (BoW, TF-IDF, embeddingi, Word2Vec)
- Model językowy (LM trenowany na nazwach polskich miejscowości)

**Function Calling** to przedostatnie/ostatnie zajęcia przed ewentualnym RAG-iem. Studenci wiedzą już jak LLM generuje tekst — teraz zobaczy jak LLM **działa** w świecie zewnętrznym.

## Cel prezentacji

Prezentacja towarzyszy warsztatowi (notebook `Function_Calling-filled_clean.ipynb`). Studenci widzą slajdy na ekranie prowadzącego **w trakcie** pracy z notebookiem — slajdy dają kontekst, motywację i wizualizacje, których nie da się dobrze pokazać w komórkach Jupytera.

Na końcu: live demo `chat_demo.py` — Gradio UI z wszystkimi narzędziami z warsztatu.

## Format

Standalone HTML (jak dotychczasowe: `Regresja Liniowa.html`, `Sieć neuronowa - standalone.html`). Generowane w Claude Design na podstawie tego planu.

---

## Struktura slajdów

### Slajd 1: Strona tytułowa

- **Tytuł:** "Function Calling — LLM jako «mózg» który wywołuje funkcje"
- **Podtytuł:** "Jak deweloperzy budują AI asystentów i agentów"
- **Kontekst:** Uczenie Maszynowe — ALK 2025/2026
- **Nastrój:** Techniczny ale przystępny. Ikona mózgu + narzędzi/kluczy

---

### Slajd 2: Gdzie jesteśmy w kursie?

- **Tytuł:** "Nasza podróż przez ML"
- **Treść:** Wizualna oś czasu / roadmapa kursu:
  1. ML Wprowadzenie
  2. Regresja liniowa
  3. Regresja logistyczna
  4. Sieci neuronowe
  5. Reprezentacja tekstu + Embeddingi
  6. Model językowy (polskie miejscowości)
  7. **→ Function Calling** ← jesteśmy tutaj
  8. (RAG?)
- **Przekaz:** "Wiemy jak LLM generuje tekst. Dziś zobaczymy jak LLM **działa** — wywołuje funkcje, odpytuje API, przeszukuje internet."
- **Dlaczego:** Studenci potrzebują zobaczyć jak FC łączy się z tym co już wiedzą. LM generuje tekst → FC daje mu "ręce" do działania w świecie.

---

### Slajd 3: Problem — LLM nie wie i nie umie

- **Tytuł:** "Co LLM umie, a czego nie umie?"
- **Treść:** Dwie kolumny:

  | LLM umie | LLM NIE umie |
  |---|---|
  | Generować płynny tekst | Sprawdzić aktualną pogodę |
  | Odpowiadać na pytania z treningu | Policzyć 17 × 23 (niezawodnie) |
  | Rozumieć kontekst | Zajrzeć do bazy danych |
  | Tłumaczyć, streszczać | Przeszukać internet |

- **Punchline:** "LLM to mózg bez rąk. Function Calling daje mu ręce."
- **Dlaczego:** Motywacja — dlaczego FC jest potrzebne. Studenci z poprzednich zajęć widzieli że LM generuje tekst, ale nie umie odpowiadać na pytania o fakty.

---

### Slajd 4: Rozwiązanie — Function Calling

- **Tytuł:** "Function Calling = LLM + narzędzia"
- **Treść:** Diagram cyklu FC (wizualizacja tego co jest w notebooku cell [13]):

  ```
  Użytkownik: "Ile to jest 17 × 23?"
       ↓
  LLM myśli: "Potrzebuję calculate('17 * 23')"    ← LLM generuje argumenty
       ↓
  Nasz kod: calculate("17 * 23") → 391             ← Python liczy
       ↓
  LLM: "17 razy 23 to 391"                          ← LLM formułuje odpowiedź
  ```

- **Kluczowy przekaz (pogrubić):** "LLM **nie liczy sam** — prosi NASZ kod o obliczenie. My kontrolujemy narzędzia."
- **Dlaczego:** To jest fundamentalny diagram warsztatu. Studenci muszą go zobaczyć na dużym ekranie zanim zaczną pisać kod.

---

### Slajd 5: Plan warsztatu

- **Tytuł:** "Co dziś zrobimy?"
- **Treść:** Numerowana lista z ikonami:
  1. **Kalkulator** — pierwsze narzędzie + pełny cykl FC
  2. **Pydantic + JSON Schema** — jak opisać narzędzie dla LLM-a
  3. **Instructor** — Structured Output (LLM odpowiada jako obiekt Pythona)
  4. **Pogoda + Prezydenci** — prawdziwe API i baza danych
  5. **LLM wybiera** — trzy narzędzia, model sam decyduje
  6. **Wikipedia + Web search** — prawdziwe źródła wiedzy
  7. **Tajemnica w danych** — fałszywe fakty i zaufanie do narzędzi
  8. **Combo: FC + Structured Output** — FactCheck
  9. **Pętla agentowa** — mini-agent decydujący sam
  10. **Chat UI** — jak ChatGPT, ale nasz! (live demo)
- **Dlaczego:** Studenci widzą mapę drogową zajęć. Mogą się zorientować ile zostało.

---

### Slajd 6: Kalkulator — pierwszy Function Call

- **Tytuł:** "Kalkulator — Twoje pierwsze narzędzie"
- **Treść:**
  - Kod funkcji `calculate()` — prosty, 5 linii
  - Strzałka → "Ale LLM nie zna tej funkcji! Musimy mu ją **opisać**."
  - Podział na dwa kroki:
    1. **Napisz funkcję** (zwykły Python)
    2. **Opisz ją dla LLM-a** (JSON Schema)
- **Wizualnie:** Blok kodu po lewej, strzałka, blok JSON Schema po prawej
- **Dlaczego:** Odpowiada sekcji 3 notebooka. Studenci zobaczą ten kod w notebooku, ale slajd daje kontekst "co teraz robimy i dlaczego".

---

### Slajd 7: JSON Schema — język opisu narzędzi

- **Tytuł:** "JSON Schema — jak LLM «widzi» narzędzia"
- **Treść:** Przykład JSON Schema kalkulatora (uproszczony):
  ```json
  {
    "name": "calculate",
    "description": "Wykonuje obliczenie matematyczne",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "Wyrażenie, np. '2+2'"
        }
      }
    }
  }
  ```
  - Strzałki/adnotacje przy polach: `name` → "Jak LLM poznaje narzędzie", `description` → "Kiedy go użyć", `parameters` → "Jakie dane podać"
- **Dlaczego:** JSON Schema to "niewidzialny bohater" FC. Studenci muszą go zrozumieć zanim zobaczą automatyzację przez Pydantic.

---

### Slajd 8: Pydantic — koniec z ręcznym JSON Schema

- **Tytuł:** "JSON Schema na dwa sposoby"
- **Treść:** Tabela porównawcza (z notebooka cell [16]):

  | | Ręcznie | Pydantic |
  |---|---|---|
  | **Jak** | Piszesz `{"type": "object", ...}` | Piszesz `class Args(BaseModel)` |
  | **Schema** | Ty go tworzysz | `model_json_schema()` generuje |
  | **Błędy** | Łatwo o literówkę | Pydantic waliduje |
  | **Kiedy** | Żeby zrozumieć pod spodem | W produkcji — zawsze |

- **Przekaz:** "Ręcznie — żeby zrozumieć. Pydantic — żeby nie zwariować."
- **Dlaczego:** Pomost między ręcznym Schema (slajd 7) a automatyzacją. Od tego momentu w notebooku używamy już tylko Pydantic.

---

### Slajd 9: Instructor — Structured Output

- **Tytuł:** "Instructor — LLM odpowiada jako obiekt Pythona"
- **Treść:**
  - Krótki przykład: definiujesz `class CityInfo(BaseModel)` → instructor wymusza odpowiedź w tym formacie
  - Podkreślić: "To NIE jest Function Calling — to drugi mechanizm"
  - Kod: `city = instructor_client.chat.completions.create(response_model=CityInfo, ...)`
  - Wynik: `city.name → "Kraków"`, `city.population_approx → 800000` — obiekt Pythona, nie tekst!
- **Dlaczego:** Sekcja 3b notebooka. Instructor jest kluczowy dla późniejszego Combo (FactCheck), trzeba go tu wprowadzić.

---

### Slajd 10: Instructor — retry z feedbackiem

- **Tytuł:** "Instructor sam naprawia błędy LLM-a"
- **Treść:** Diagram 3-krokowy:
  1. LLM zwraca niepoprawny JSON
  2. Instructor łapie błąd i odsyła do LLM: "Popraw, oto co było źle: ..."
  3. LLM poprawia się → poprawny obiekt Pydantic
- **Przekaz:** "W notebooku celowo sabotujemy odpowiedź — i patrzymy jak instructor ją naprawia"
- **Dlaczego:** To jest demo z cell [26] — sabotaż JSON-a i automatyczna naprawa. Warto pokazać schemat przed uruchomieniem kodu.

---

### Slajd 11: FC vs Structured Output — dwa różne mechanizmy

- **Tytuł:** "Function Calling vs. Structured Output"
- **Treść:** Tabela (z notebooka cell [29]):

  | | **Function Calling** | **Structured Output** |
  |---|---|---|
  | **Co robi LLM** | **Wybiera** funkcję do wywołania | **Wypełnia** schemat danymi |
  | **Kto wykonuje pracę** | Nasz kod Python | Sam LLM |
  | **Parametr** | `tools=tools_definition` | `response_model=MyModel` |
  | **Wynik** | LLM woła funkcję → wynik → odpowiedź | Obiekt Pydantic |

- **Analogia kuchenna:**
  - **FC** = LLM jest **kelnerem** — przyjmuje zamówienie i mówi kucharzowi (naszemu kodowi) co ugotować
  - **SO** = LLM jest **kucharzem** — musi podać danie na określonym talerzu (Pydantic model)
- **Dlaczego:** Kluczowe rozróżnienie. Studenci często mylą te dwa mechanizmy. Analogia kuchenna pomaga zapamiętać.

---

### Slajd 12: Pogoda — prawdziwe API

- **Tytuł:** "Pogoda — pierwsze prawdziwe API"
- **Treść:**
  - Schemat: `LLM → get_weather("Kraków") → wttr.in API → "18°C, słonecznie"` → LLM formułuje odpowiedź
  - Nowość vs kalkulator: teraz funkcja **łączy się z internetem** (API wttr.in)
  - Fallback: jeśli API nie działa → dane zastępcze (mock)
- **Przekaz:** "Kalkulator był zabawką. Pogoda to prawdziwe narzędzie — łączy się z zewnętrznym API."
- **Dlaczego:** Skok jakościowy — od zabawki do czegoś użytecznego. Studenci widzą że FC to nie tylko "wywoływanie funkcji" ale prawdziwa integracja z światem.

---

### Slajd 13: Prezydenci + trzy narzędzia

- **Tytuł:** "Trzy narzędzia — LLM sam wybiera!"
- **Treść:**
  - Wizualizacja: LLM na środku, trzy strzałki do narzędzi:
    - 🌤️ `get_weather` → API pogody
    - 🔢 `calculate` → kalkulator
    - 🏛️ `search_presidents` → baza prezydentów
  - Pytania testowe (z notebooka cell [38]):
    - "Jaka jest pogoda w Gdańsku?" → `get_weather`
    - "Ile to jest 17 razy 23?" → `calculate`
    - "Kto był najmłodszym prezydentem?" → `search_presidents`
    - "Co to jest Python?" → żadne narzędzie!
- **Przekaz:** "LLM sam decyduje KTÓRE narzędzie wywołać — albo żadne."
- **Dlaczego:** Kluczowy moment — do tej pory było 1 narzędzie, teraz LLM musi **wybierać**. To jest serce FC.

---

### Slajd 14: Ćwiczenie 1 — Nazwa vs Opis narzędzia

- **Tytuł:** "Co ważniejsze — nazwa czy opis?"
- **Treść:**
  - **Część A:** Zmieniamy opis `get_weather` na mylący → czy LLM się pomyli?
  - **Część B — Cichy błąd:** Ukrywamy nazwy (`tool_alpha`, `tool_beta`) i zamieniamy opisy → LLM wywołuje ZŁĄ funkcję, ale wynik "wygląda OK"
  - Ostrzeżenie (czerwony box): "Cichy błąd jest GORSZY od crashu! Nie ma Error — a dane są złe."
- **Przekaz:** "Nazwy i opisy narzędzi to krytyczny element bezpieczeństwa systemów AI."
- **Dlaczego:** Ćwiczenie 1 jest kluczowe dydaktycznie. Studenci odkrywają że nazwa > opis, i że cichy błąd jest groźniejszy niż crash. Slajd na ekranie prowadzącego daje instrukcję.

---

### Slajd 15: Ćwiczenie 2 — Dodaj własne narzędzie

- **Tytuł:** "Ćwiczenie: Dodaj get_population"
- **Treść:**
  - Zadanie: Napisz funkcję `get_population(city)` + model Pydantic + `make_tool()` + dodaj do `AVAILABLE_TOOLS`
  - Mini-tabelka z danymi GUS (Warszawa ~1.86M, Kraków ~800K, ...)
  - Wzorzec: "Wzoruj się na `get_weather`"
- **Dlaczego:** Slajd-instrukcja widoczna na ekranie prowadzącego gdy studenci pracują.

---

### Slajd 16: Wikipedia i Web Search

- **Tytuł:** "Prawdziwe źródła wiedzy — Wikipedia + DuckDuckGo"
- **Treść:**
  - Dwa nowe narzędzia:
    - 📚 `search_wikipedia` — encyklopedyczna wiedza (biblioteka `wikipedia`)
    - 🔍 `search_web` — aktualne wydarzenia (DuckDuckGo, bez klucza API!)
  - Kiedy które?
    - Wikipedia: "Kto to był Nikola Tesla?" (fakty historyczne)
    - DuckDuckGo: "Aktualny kurs dolara" (dane bieżące)
  - "Dlaczego DuckDuckGo a nie Google?" → brak klucza API, darmowe
- **Dlaczego:** Dwa kolejne narzędzia dodawane w sekcjach 7-8 notebooka. Slajd daje motywację i rozróżnienie.

---

### Slajd 17: Ćwiczenie 3 — Zbuduj narzędzie Wikipedia

- **Tytuł:** "Ćwiczenie: Narzędzie Wikipedia"
- **Treść:**
  - Zadanie: Napisz `search_wikipedia(query)` używając biblioteki `wikipedia`
  - API reference: `wikipedia.search()` → lista tytułów, `wikipedia.page()` → obiekt z `.title`, `.summary`, `.url`
  - Uwaga: `DisambiguationError` — co to jest i jak obsłużyć
- **Dlaczego:** Instrukcja do ćwiczenia 3. Wikipedia to pierwsze narzędzie sięgające do internetu.

---

### Slajd 18: Tajemnica w danych — fałszywe fakty

- **Tytuł:** "Tajemnica w danych — czy narzędziom można ufać?"
- **Treść:**
  - Ujawnienie: "W pliku `prezydenci_polski.md` celowo ukryłem dwa zmyślone fakty"
  - Przykłady fałszywych faktów (np. "popiersie Kleopatry od ambasadora Egiptu", "kolekcja antycznych monet")
  - Agent zwrócił je **z pełnym przekonaniem** — bo nasze narzędzie mu je podało
  - Tabela zagrożeń:

    | Źródło błędu | Efekt | Wykrywalność |
    |---|---|---|
    | Zły opis narzędzia | LLM i tak trafia (nazwa!) | Łatwa |
    | Podmienione opisy + nazwy | Cichy błąd | Żadna |
    | **Fałszywe dane w narzędziu** | **LLM powtarza z przekonaniem** | **Wymaga weryfikacji** |

- **Przekaz:** "Garbage in, garbage out. Ale szansa: jeśli damy agentowi narzędzie do weryfikacji..."
- **Dlaczego:** Ważna lekcja o zaufaniu do danych. Naturalnie prowadzi do combo FC + FactCheck.

---

### Slajd 19: Combo — FC + Structured Output = FactCheck

- **Tytuł:** "Combo: Function Calling + Structured Output"
- **Treść:**
  - Wzorzec ETL z LLM-em:
    1. **Extract** (FC): `search_presidents`, `search_wikipedia`, `search_web` — zbierz dowody
    2. **Transform** (Instructor): LLM analizuje dowody i wypełnia `FactCheck`
    3. **Load**: Gotowy obiekt Pydantic z polami `claim`, `evidence`, `verdict`, `confidence`
  - Schemat wizualny: Narzędzia → nieustrukturyzowany tekst → Instructor → FactCheck(verdict="fałsz", confidence=0.9)
- **Przekaz:** "FC zbiera dane, Instructor je formatuje. Dwa mechanizmy — jedna siła."
- **Kiedy stosować:** Gdy źródło zwraca nieustrukturyzowany tekst, a Ty potrzebujesz konkretnych pól.
- **Dlaczego:** To kulminacja warsztatu — oba mechanizmy (FC i SO) pracują razem.

---

### Slajd 20: Ćwiczenie 4 — Model FactCheck

- **Tytuł:** "Ćwiczenie: Uzupełnij model FactCheck"
- **Treść:**
  - Zadanie: Uzupełnij 5 pól Pydantic:
    - `claim: str` — sprawdzane twierdzenie
    - `evidence: str` — znalezione dowody
    - `verdict: Literal["prawda", "fałsz", "nie da się zweryfikować"]`
    - `confidence: float` — pewność 0-1 (z `Field(ge=0, le=1)`)
    - `reasoning: str` — uzasadnienie werdyktu
  - Twierdzenia do testowania: Nobel Wałęsy, fałszywe fakty o Kwaśniewskim, Dudzie
- **Dlaczego:** Instrukcja do ćwiczenia 4 na ekranie prowadzącego.

---

### Slajd 21: Problem snippetów → fetch_webpage

- **Tytuł:** "Problem: snippety nie wystarczą"
- **Treść:**
  - `search_web` zwraca krótkie fragmenty: *"Pociąg wyróżnia się jako dobry środek tra..."*
  - LLM musi **zgadywać** resztę → ogólnikowe odpowiedzi
  - Rozwiązanie: `fetch_webpage(url)` — agent najpierw szuka (DDG), potem **czyta** wybraną stronę
  - Analogia: "Jak człowiek — szukasz w Google, klikasz link, czytasz artykuł"
- **Dlaczego:** Motywacja do dodania `fetch_webpage` przed pętlą agentową. Bez tego narzędzia agent nie potrafi czytać stron.

---

### Slajd 22: Pętla agentowa — LLM który myśli i działa

- **Tytuł:** "Pętla agentowa — od jednego narzędzia do wielu"
- **Treść:**
  - Do tej pory: jedno pytanie → jedno narzędzie → odpowiedź
  - Teraz: pytanie może wymagać **kilku** narzędzi po kolei
  - Pseudokod pętli (z notebooka cell [93]):
    ```
    while LLM chce wywołać narzędzie:
        1. LLM wybiera narzędzie i argumenty
        2. Nasz kod wywołuje funkcję
        3. Wynik wraca do LLM-a
        4. LLM decyduje:
             potrzebuję jeszcze? → wróć do 1.
             mam wszystko?      → generuję odpowiedź
    ```
  - Przykład: "Sprawdź w bazie ile lat miał Kwaśniewski gdy został prezydentem i oblicz ile to dni" → search_presidents → calculate
- **Przekaz:** "To jest mini-agent. Sam decyduje co wywołać i kiedy zakończyć."
- **Dlaczego:** Kluczowy skok konceptualny — od single-shot FC do pętli agentowej. Pseudokod na dużym ekranie pomaga zrozumieć mechanizm.

---

### Slajd 23: Wszystkie narzędzia agenta

- **Tytuł:** "Arsenał naszego agenta"
- **Treść:** Wizualna mapa wszystkich 7 narzędzi:

  | Narzędzie | Źródło | Typ |
  |---|---|---|
  | `calculate` | Python eval | Obliczenia |
  | `get_weather` | wttr.in API | Dane bieżące |
  | `search_presidents` | prezydenci_polski.md | Baza lokalna |
  | `get_population` | słownik Python | Dane statyczne |
  | `search_wikipedia` | Wikipedia API | Encyklopedia |
  | `search_web` | DuckDuckGo | Internet |
  | `fetch_webpage` | HTTP GET | Czytanie stron |

- **Przekaz:** "Od jednego kalkulatora do pełnego zestawu — zbudowaliśmy to krok po kroku."
- **Dlaczego:** Moment "wow" — studenci widzą ile narzędzi już mają. Podsumowanie przed ćwiczeniami z agentem.

---

### Slajd 24: Ćwiczenie 5 — Trudne pytanie do agenta

- **Tytuł:** "Ćwiczenie: Zadaj agentowi trudne pytanie"
- **Treść:**
  - Zadanie: Wymyśl pytanie wymagające **2+ narzędzi**
  - Inspiracje:
    - "Sprawdź na Wikipedii kto wynalazł telefon i oblicz ile lat temu to było"
    - "Jaka pogoda w Krakowie i ile osób tam mieszka? Oblicz mieszkańców na stopień"
    - "Znajdź na Wikipedii kiedy zbudowano Wawel i sprawdź aktualną pogodę w Krakowie"
  - Obserwuj: jak agent krok po kroku dochodzi do odpowiedzi
- **Dlaczego:** Instrukcja do ćwiczenia 5.

---

### Slajd 25: Ćwiczenie 6 (bonus) — Twój asystent

- **Tytuł:** "Bonus: Zbuduj własnego asystenta"
- **Treść:**
  - Tabelka pomysłów:

    | Temat | Narzędzia |
    |---|---|
    | Podróżniczy | pogoda + Wikipedia + kalkulator (waluty) |
    | Filmowy | Wikipedia + web search + kalkulator (oceny) |
    | Naukowy | Wikipedia + kalkulator + web search |
    | Kulinarny | web search + Wikipedia (kuchnie świata) |

  - "Zmień SYSTEM_PROMPT, dodaj/usuń narzędzia, zadaj pytanie!"
- **Dlaczego:** Bonus dla szybszych studentów. Daje wolność twórczą.

---

### Slajd 26: Chat z pamięcią konwersacji

- **Tytuł:** "Chat z pamięcią — multi-turn"
- **Treść:**
  - Problem: `ask_with_tools()` nie pamięta kontekstu
  - Przykład dialogu:
    ```
    Ty:  "Jaka pogoda we Wrocławiu?"
    LLM: "15°C, częściowe zachmurzenie"
    Ty:  "A jaka populacja tego miasta?"
         ↑ LLM musi pamiętać że chodzi o Wrocław!
    ```
  - Rozwiązanie: `messages` żyje poza funkcją — każde pytanie widzi poprzednie
  - Jedna zmiana: `chat_with_tools(messages, question)` — messages to lista modyfikowana in-place
- **Dlaczego:** Most między notebook a live demo. Studenci widzą jak dodać pamięć do agenta.

---

### Slajd 27: Podsumowanie — co zrobiliśmy

- **Tytuł:** "Co dziś zbudowaliśmy?"
- **Treść:** Lista z podsumowaniem (z notebooka cell [107]):
  1. **Pydantic + Instructor** — JSON Schema, walidacja, LLM jako obiekt Pythona
  2. **FC vs SO** — dwa mechanizmy, dwa klienty, komplementarne role
  3. **7 narzędzi** — od kalkulatora po web search
  4. **Pod maską** — co LLM widzi, jak wybiera, cichy błąd
  5. **Combo** — FC zbiera dane → SO formatuje werdykt
  6. **Pętla agentowa** — mini-agent z wieloma narzędziami
  7. **Chat z pamięcią** — multi-turn konwersacja
- **Dlaczego:** Recap przed live demo.

---

### Slajd 28: Tak działają prawdziwe produkty AI

- **Tytuł:** "To nie jest zabawka — tak działają prawdziwe produkty"
- **Treść:** Tabela (z notebooka cell [107]):

  | Produkt | Narzędzia (FC) | Structured Output |
  |---|---|---|
  | ChatGPT | Przeglądarka, DALL-E, Code Interpreter | Wewnętrzne parsowanie |
  | GitHub Copilot | Czytanie plików, terminal, wyszukiwarka | Ustrukturyzowany diff |
  | Claude | Computer use, MCP, web search | Structured responses |
  | Cursor | Edycja kodu, lint, testy | Diff format |

- **Przekaz:** "Wasz notebook to miniaturowa wersja tego samego wzorca."
- **Dlaczego:** Kontekst przemysłowy. Studenci widzą że to co zrobili to prawdziwy pattern.

---

### Slajd 29: Architektura — jak to się składa

- **Tytuł:** "Od notebooka do aplikacji"
- **Treść:** Diagram 3-warstwowy:
  1. **Frontend** = Gradio / Streamlit / React (interfejs użytkownika)
  2. **Backend** = Function Calling + pętla agentowa (to co pisaliśmy)
  3. **Narzędzia** = API pogody, Wikipedia, DuckDuckGo, bazy danych...
- **Przekaz:** "Jednej linijki brakuje do pełnej aplikacji AI..."
- **Dlaczego:** Przygotowanie do live demo. Studenci widzą architekturę zanim zobaczą Gradio UI.

---

### Slajd 30: Live Demo — Chat UI

- **Tytuł:** "🚀 Live Demo: Nasz ChatGPT"
- **Treść:**
  - Screenshot / wireframe interfejsu Gradio z:
    - 💬 Dymki wiadomości (user / assistant)
    - 🧠 Collapsible reasoning (tok myślenia)
    - 🔧 Tool calls z wynikami
    - 📝 Pamięć konwersacji (multi-turn)
    - 🔄 Agent loop
  - "Pod spodem: dokładnie ten sam kod co pisaliśmy — opakowany w Gradio"
  - Instrukcja uruchomienia: `python chat_demo.py` lub z notebooka sekcja 13
- **Dlaczego:** Slajd wprowadzający do live demo. Prowadzący w tym momencie przełącza się na `chat_demo.py`.

---

### Slajd 31: Co dalej? RAG i beyond

- **Tytuł:** "Co dalej?"
- **Treść:**
  - Bridge slide do potencjalnego RAG-a:
    - Dziś: narzędzia z gotowymi danymi (Wikipedia, baza prezydentów, web)
    - Następnie: **RAG** — podłączanie **własnych dokumentów** (PDF, bazy wiedzy, embeddingi!)
    - "Pamiętacie embeddingi z zajęć o reprezentacji tekstu? Wracają!"
  - Oś ewolucji:
    ```
    Function Calling (dziś) → RAG (własne dokumenty) → pełne agenty
    ```
- **Dlaczego:** Zamknięcie łuku kursu. FC → RAG to naturalny następny krok. Embeddingi łączą te zajęcia z poprzednimi.

---

## Proponowana kolejność slajdów — podsumowanie

| # | Tytuł | Typ | Sekcja notebooka |
|---|---|---|---|
| 1 | Strona tytułowa | Intro | — |
| 2 | Gdzie jesteśmy w kursie? | Kontekst | — |
| 3 | Co LLM umie, a czego nie umie? | Motywacja | — |
| 4 | Function Calling = LLM + narzędzia | Diagram FC | cell [13] |
| 5 | Plan warsztatu | Agenda | cell [0] |
| 6 | Kalkulator — pierwsze narzędzie | Kod | sekcja 3 |
| 7 | JSON Schema — język opisu narzędzi | Teoria | cell [12] |
| 8 | Pydantic — koniec z ręcznym Schema | Porównanie | cell [16] |
| 9 | Instructor — Structured Output | Teoria | sekcja 3b |
| 10 | Instructor — retry z feedbackiem | Diagram | cell [25-26] |
| 11 | FC vs Structured Output | Porównanie | cell [29] |
| 12 | Pogoda — prawdziwe API | Narzędzie | sekcja 4 |
| 13 | Trzy narzędzia — LLM wybiera | Wizualizacja | sekcja 6 |
| 14 | Ćwiczenie 1 — Nazwa vs Opis | Ćwiczenie | sekcja 6 ćw. |
| 15 | Ćwiczenie 2 — get_population | Ćwiczenie | sekcja 6 ćw. |
| 16 | Wikipedia + DuckDuckGo | Narzędzia | sekcje 7-8 |
| 17 | Ćwiczenie 3 — Wikipedia | Ćwiczenie | sekcja 7 ćw. |
| 18 | Tajemnica w danych | Bezpieczeństwo | sekcja 8b |
| 19 | Combo: FC + SO = FactCheck | Wzorzec | sekcja 9 |
| 20 | Ćwiczenie 4 — FactCheck | Ćwiczenie | sekcja 9 ćw. |
| 21 | Problem snippetów → fetch_webpage | Motywacja | sekcja pre-10 |
| 22 | Pętla agentowa | Diagram | sekcja 10 |
| 23 | Arsenał naszego agenta | Podsumowanie | — |
| 24 | Ćwiczenie 5 — Trudne pytanie | Ćwiczenie | sekcja 10 ćw. |
| 25 | Ćwiczenie 6 (bonus) — Asystent | Ćwiczenie | sekcja 10 ćw. |
| 26 | Chat z pamięcią | Teoria | sekcja 12 |
| 27 | Co dziś zbudowaliśmy? | Recap | sekcja 11 |
| 28 | Prawdziwe produkty AI | Kontekst | cell [107] |
| 29 | Od notebooka do aplikacji | Architektura | sekcja 13 |
| 30 | Live Demo: Nasz ChatGPT | Demo | chat_demo.py |
| 31 | Co dalej? RAG | Zamknięcie | — |

**Razem: 31 slajdów** (20 treściowych + 6 ćwiczeniowych + 5 intro/outro)

---

## Wskazówki dla Claude Design

### Styl wizualny
- Spójny z poprzednimi prezentacjami (Regresja Liniowa, Sieć Neuronowa)
- Kolorowe boxy jak w notebooku: zielony (sukces/ważne), żółty (uwaga/porównanie), niebieski (info), czerwony (ostrzeżenie/błąd)
- Dużo diagramów i schematów — mniej tekstu, więcej wizualizacji
- Kod na slajdach: krótki, z kolorowaniem składni, max 8-10 linii

### Elementy interaktywne (na slajdach ćwiczeniowych)
- Wyraźna instrukcja: "Otwórz notebook, sekcja X"
- Timer / wskazówka czasowa: "~5 minut"
- Podpowiedź ukryta (albo tekst "podpowiedź w notebooku")

### Kluczowe wizualizacje do zaprojektowania
1. **Slajd 4:** Diagram cyklu FC (strzałki, ikony) — najważniejszy diagram prezentacji
2. **Slajd 11:** Tabela FC vs SO z analogią kuchenną (ikony kelner/kucharz)
3. **Slajd 13:** Radialna mapa "LLM na środku, narzędzia dookoła"
4. **Slajd 19:** Pipeline ETL: Extract → Transform → Load z ikonami
5. **Slajd 22:** Flowchart pętli agentowej (while loop z decyzjami)
6. **Slajd 23:** Tabela wszystkich 7 narzędzi z ikonami
7. **Slajd 29:** 3-warstwowy diagram architektury (frontend/backend/narzędzia)

### Język
- Polski
- Luźny, warsztatowy ton (nie akademicki)
- "Ty/Twój" (bezpośredni zwrot do studenta)
