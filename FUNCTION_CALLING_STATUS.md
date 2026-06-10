# Function Calling — status prac (aktualizowany przez Claude)

**Branch:** `docker-uv-setup`
**Ostatnia aktualizacja:** 2026-05-10

## Co to jest

Warsztat Function Calling dla studentów ALK — notebook `Function_Calling.ipynb` (107 komórek, 14 sekcji).
Notebook jest generowany z `_build_fc_notebook.py` (jedyne źródło prawdy — edytujemy builder, nie notebook).

**WAŻNE:** Każda komórka w notebooku MUSI mieć swoje źródło w builderze.
Jeśli komórka istnieje tylko w .ipynb (dodana ręcznie) — builder ją nadpisze przy przebudowie!

## Architektura

| Plik | Rola |
|---|---|
| `_build_fc_notebook.py` | Builder — generuje `Function_Calling.ipynb` |
| `utils.py` | Helpery: `connect_llm()`, `extract_reasoning()`, `clean_content()`, `print_reasoning()`, `ensure_package()` |
| `chat_ui.py` | Gradio Chat UI (sekcja 13) — import z notebooka i standalone |
| `chat_demo.py` | Standalone chat demo (CLI: `--backend`, `--api-key`, `--model`, `--port`) |
| `przed_zajeciami.py` | Przygotowanie notebooków dla studentów (pre-push hook) |
| `prezydenci_polski.md` | Baza danych narzędzia `search_presidents` (z celowo zmyślonymi faktami) |

## Struktura notebooka (sekcje)

1. Konfiguracja środowiska
2. Połączenie z LLM-em (`connect_llm` + `API_KEY` + zakomentowane `model=`/`backend=`)
2b. Komórka "Nadpisanie modelu" — prowadzący może szybko przełączyć model/backend
2c. `DEFAULT_SYSTEM_PROMPT` — centralny system prompt, warunkowy `<|think|>` dla Gemma-4
3. Kalkulator — pierwsze narzędzie + FC; opcjonalnie instructor demo (3b)
4. Pogoda — `get_weather()` z wttr.in + fallback na mock
5. Baza prezydentów — `search_presidents()` z `prezydenci_polski.md`
6. Trzy narzędzia — LLM wybiera; Ćw.1A (nazwy vs opisy), Ćw.1B (cichy błąd) + wnioski
7. Wikipedia — `search_wikipedia()`; Ćw.3
8. Web search — `search_web()` przez DuckDuckGo (ddgs)
8b. Tajemnica w danych — zagrożenie i szansa (3 kolorowe divy: czerwony/żółty/zielony)
9. Combo FC + Structured Output; `verify_claim()` + Ćw.4 FactCheck + info box ETL
10. Pętla agentowa (`agent()`); Ćw.5 (trudne pytanie), Ćw.6 (własny asystent)
11. Podsumowanie + tabela produktów AI + kluczowe wnioski
12. Bonus: `chat_with_tools()` — multi-turn z pamięcią konwersacji
13. Bonus: Chat UI (Gradio) — `launch_chat()` z `chat_ui.py` + hasło prowadzącego

## Kluczowe decyzje techniczne

- **DEFAULT_SYSTEM_PROMPT:** Zdefiniowany RAZ w Cell 7, używany we WSZYSTKICH funkcjach.
  `<|think|>` dodawany warunkowo: `_is_gemma = "gemma" in MODEL_NAME.lower()`.
  Inne modele (Qwen3, Llama) dostają prompt bez tego prefixu.
- **Gemma-4 natywny reasoning:** Model używa tokenów `<|channel>thought ... <channel|>` w `msg.content`.
  `extract_reasoning()` w `utils.py` wyciąga je automatycznie (obok Qwen3 `reasoning_content`).
  `clean_content()` zwraca treść bez artefaktów.
- **connect_llm(lecturer_server, model, api_key, backend):**
  - `model=` — partial match na nazwie (zakomentowany domyślnie, `przed_zajeciami.py` komentuje)
  - `api_key=` — opcjonalny klucz do LM Studio z Require Authentication
  - `backend="ollama"/"lmstudio"` — pomiń auto-detection, idź prosto do wybranego backendu
  - `przed_zajeciami.py` komentuje `model=`/`backend=` i resetuje `API_KEY=None`
- **chat_demo.py:** Standalone CLI z flagami `--backend`, `--api-key`, `--model`, `--port`.
  Warunkowy `<|think|>` (jak w notebooku). Animacja oczekiwania + detekcja kolejki w Gradio UI.
- **Sekcja 8b — fikcyjne fakty:** `prezydenci_polski.md` zawiera 2 zmyślone "mało znane fakty"
  (popiersie Kleopatry, monety Cezara). Użyte dydaktycznie: agent zwraca je z pełnym przekonaniem.
  3 kolorowe divy: czerwony (zagrożenie), żółty (tabela wykrywalności), zielony (szansa — know-how firmy).
- **pre-push hook:** Sanityzuje LECTURER_SERVER (prawdziwy IP → placeholder), odpala `przed_zajeciami.py`.
  Komentuje `model=`/`backend=`, resetuje `API_KEY`, zwija Rozwiązania/Podpowiedzi.

## Co zostało zrobione

### Sesja 2026-05-10 (wieczór) — przywracanie notebooka po crash'u

- [x] Porównanie `git stash@{0}` (złoty stan sprzed crash'a) z aktualnym builderem
- [x] Dodano sekcję 13: Chat UI — Gradio z `launch_chat()`, hasłem, info boxem
- [x] Dodano komórkę "Nadpisanie modelu" (Cell 6) — prowadzący szybko przełącza model
- [x] Dodano "Wnioski z Ćwiczenia 1" — tabela: źródło błędu → efekt → wykrywalność
- [x] Dodano info box ETL po FactCheck (wzorzec Extract → Transform → Load)
- [x] Wydzielono test FactCheck do osobnej komórki (separator + komórka test + info box)
- [x] Usunięto sekcję "Bonus: FC + Structured Output w pipeline" (nie było w stashu)
- [x] Przebudowano notebook — struktura sekcji identyczna ze stashem (14 sekcji: 1–8, 8b, 9–13)

### Sesja 2026-05-10 (rano) — api_key, backend, animacja, <|think|>

- [x] `connect_llm`: nowy parametr `api_key=` dla LM Studio z Require Authentication
- [x] `connect_llm`: nowy parametr `backend=` — pomiń LM Studio/Ollama w auto-detection
- [x] `detect_lmstudio`: obsługuje Bearer token (auth header) przy listowaniu modeli
- [x] `chat_ui.py`: animacja oczekiwania (ThreadPoolExecutor) + detekcja kolejki (>8s)
- [x] `chat_demo.py`: flagi CLI `--backend`, `--api-key`, `--model`, `--port`
- [x] DEFAULT_SYSTEM_PROMPT: warunkowy `<|think|>` tylko dla Gemma (`_is_gemma`)
- [x] `przed_zajeciami.py`: komentuje `model=`/`backend=`, resetuje `API_KEY`, `import re` na górze
- [x] Scentralizowany system prompt — 7 inline kopii → 1 `DEFAULT_SYSTEM_PROMPT`

### Sesja wcześniejsza — reasoning, clean_content, chat UI

- [x] `utils.py`: `extract_reasoning()` obsługuje Gemma-4 channel tokens + Qwen3
- [x] `utils.py`: nowa `clean_content()` — czyści artefakty reasoning z msg.content
- [x] `_build_fc_notebook.py`: wszystkie funkcje używają `print_reasoning()` + `clean_content()`
- [x] `ask_with_tools`: natywny reasoning → fallback na instructor ToolReasoning
- [x] Sekcja 8b: 3 kolorowe divy (zagrożenie/tabela/szansa) z treścią ze stasha
- [x] Usunięto "garbage in garbage out" i "tajemnica prezydencka" (zastąpione przez 8b)

## TODO / pomysły na przyszłość

- [ ] Przetestować na żywo natywny reasoning Gemmy-4 (`<|channel>thought` z `<|think|>`)
- [ ] Rozważyć dodanie narzędzia `get_exchange_rate()` (kurs walut)
- [ ] Przetestować sekcję 13 Chat UI z serwerem prowadzącego (hasło "chat")
- [ ] Merge docker-uv-setup → main (po przetestowaniu na zajęciach)
- [ ] Oznaczyć nowe komórki (Nadpisanie modelu, test FactCheck) jako student-stub jeśli trzeba
