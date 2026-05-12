# RAG Workshop Notebook Redesign

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign RAG_warsztat.ipynb with a natural bridge from Function Calling (substring → context stuffing → embeddings/RAG), keeping prezydenci_polski as dataset, and adding a proper 3-method comparison.

**Architecture:** The notebook starts where FC left off — showing substring search limitations, then introduces context stuffing (LLM reads entire document) as a stepping stone, then builds full RAG pipeline. The 3-method comparison table at the end crystallizes why embeddings win at scale. All inline code — no new utils.py functions. Same `connect_llm()` + `setup_auth_client()` setup as FC notebook.

**Tech Stack:** sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2), openai (via Ollama/LM Studio), numpy, matplotlib, sklearn (t-SNE)

---

## Design Rationale

### Why smart_search as RAG opener?

In Function_Calling, students built `search_presidents()` — a substring matcher. They saw it fail on synonyms ("związkowiec" ≠ "NSZZ Solidarność"), periphrases ("międzynarodowa nagroda" ≠ "Nobel"), and semantic queries ("kto zbierał monety"). The smart_search code was removed from FC and archived in `_rag_smart_search_notes.py` precisely for reuse here.

The RAG notebook should open with: "Pamiętacie te ograniczenia?" → show 3 approaches side-by-side:

| Metoda | Jak działa | Plusy | Minusy |
|---|---|---|---|
| **Substring** | `if query in text` | Szybkie, proste | Nie rozumie synonimów, odmian |
| **Context stuffing** | Cały tekst → LLM | Rozumie język | Nie skaluje się (limit tokenów) |
| **RAG (embeddingi)** | Similarity → top-K → LLM | Skaluje się + rozumie | Wymaga modelu embeddingowego |

### Why stay with prezydenci_polski?

1. **Continuity** — students already know this data from FC workshop
2. **Verifiability** — students can fact-check RAG answers (they know the presidents)
3. **Right size** — small enough for context stuffing demo (showing it works), but the limitation is obvious ("co gdy mamy 1000 stron?")
4. **Polish language** — tests embedding model's multilingual capabilities

Wolne Lektury would be a bad choice: students can't verify answers about literary content they may not know, and the questions would be vague. The existing `prezydenci_polski.txt` (55 lines, ~4K chars) is pedagogically ideal.

### What to borrow from kjedrzejewski's course

- **Multi-query RAG** (008-01): LLM generates multiple search queries from one question → better recall. Good as an advanced exercise.
- **Comparison structure**: the side-by-side demo pattern (without RAG vs with RAG) is already in our notebook and works well.
- **Skip**: Weaviate (too much infra), FAISS (numpy is fine for this scale), data generation notebooks (not needed).

### What's changing vs current notebook

| Current | New |
|---|---|
| Starts cold with "LLM nie zna odpowiedzi" | Starts with FC bridge: "substring search miał limity" |
| No context stuffing | Section 3: context stuffing as stepping stone |
| 3 exercises (chunk size, cosine, prompt) | 4 exercises (+1: implement context stuffing) |
| Inline `chunk_text`, `search`, `build_rag_prompt` | Same — keep inline for visibility |
| prezydenci_polski.txt only | Same dataset, better motivation |
| t-SNE at end | Keep t-SNE, add comparison table |

---

## Notebook Outline (target: ~38 cells)

```
Section 0: Intro + Plan                    (2 cells: md, md)
Section 1: Konfiguracja                    (4 cells: md, md, code-colab, code-imports)
Section 2: Połączenie z LLM               (2 cells: md, code-connect_llm)
Section 3: Most z FC — limity substringa   (3 cells: md, code-substring, code-demo-fail)
Section 4: Context stuffing                (3 cells: md, code-stuffing, md-exercise1)
           Ćwiczenie 1                     (1 cell: code-exercise1)
Section 5: Tabela porównawcza              (1 cell: md-table)
Section 6: Chunking                        (3 cells: md, code-load+chunk, md-exercise2)
           Ćwiczenie 2                     (1 cell: code-exercise2)
Section 7: Embeddingi                      (3 cells: md, code-load-model, code-encode)
Section 8: Cosine similarity               (2 cells: md-exercise3, code-exercise3)
Section 9: Wyszukiwanie semantyczne        (3 cells: md, code-search-fn, code-demo)
Section 10: RAG prompt                     (3 cells: md, code-build-prompt, code-rag-answer)
Section 11: Porównanie BEZ vs Z RAG        (2 cells: md, code-comparison)
Section 12: Ćwiczenie 4 — prompt RAG       (2 cells: md-exercise4, code-exercise4)
Section 13: Zadaj pytanie                  (2 cells: md, code-interactive)
Section 14: Wizualizacja                   (2 cells: md, code-tsne)
Section 15: Podsumowanie                   (1 cell: md)
                                    Total: ~38 cells
```

---

## File Structure

```
Files to modify:
  RAG_warsztat.ipynb          — full rewrite of notebook cells

Files to read (not modify):
  _rag_smart_search_notes.py  — source for context stuffing code + test queries
  Function_Calling.ipynb      — reference for search_presidents() + setup pattern
  prezydenci_polski_biografie.txt — RAG data source (738-line biographies, new)
  prezydenci_polski.md        — context stuffing data source (structured, used by FC)
  utils.py                    — connect_llm, setup_auth_client, ensure_package
```

---

## Tasks

### Task 1: Sections 0–2 — Intro, Config, LLM Connection

**Files:**
- Modify: `RAG_warsztat.ipynb` (cells 0–7)

These sections are mostly done from the previous alignment work. The key change is **Cell 0** — update the intro to reference FC and add the 3-method roadmap.

- [ ] **Step 1: Rewrite Cell 0 (intro markdown)**

Replace the current intro with one that bridges from FC:

```markdown
# RAG — Retrieval Augmented Generation

## Jak sprawić, żeby LLM "wiedział" o rzeczach, których nie ma w jego danych treningowych?

### Skąd przychodzimy?

W warsztacie **Function Calling** zbudowaliśmy `search_presidents()` — wyszukiwarkę po prezydentach Polski.
Działała na dopasowaniu słów kluczowych. Widzieliśmy, że pytania typu *"kto zbierał monety"* lub *"związkowiec"* ją łamią.

Dziś zobaczymy **trzy podejścia** do tego problemu:

| # | Metoda | Jak działa | Skaluje się? |
|---|---|---|---|
| 1 | **Substring search** | `if query in text` | ✓ ale nie rozumie języka |
| 2 | **Context stuffing** | Cały tekst → do LLM-a | ✗ limit tokenów |
| 3 | **RAG (embeddingi)** | Wyszukaj top-K → podaj LLM-owi | ✓ + rozumie język |

### Plan warsztatu

1. **Pokażemy problem**: substring search nie rozumie synonimów
2. **Context stuffing**: wpychamy cały dokument do LLM-a — działa, ale nie skaluje się
3. **Embeddingi**: zamienimy tekst na wektory i pokażemy wyszukiwanie semantyczne
4. **RAG**: skleimy wyszukiwanie z LLM-em — i zobaczymy różnicę!
```

- [ ] **Step 2: Verify cells 1–7 are correct (config, Colab, imports, LLM connection)**

These were set up in previous commits. Verify the cell sequence is:
1. `## 1. Konfiguracja środowiska` (md)
2. Colab hint (md)
3. Colab download code — commented (code)
4. Imports: `ensure_package`, sentence-transformers, openai, numpy, textwrap (code)
5. `## 2. Połączenie z LLM-em` (md with instructions box)
6. `connect_llm()` + `setup_auth_client()` (code)

No changes needed here unless cell order drifted.

- [ ] **Step 3: Run cells 0–7 in Jupyter to verify they execute**

Run: open notebook in Jupyter, execute cells 0 through the connect_llm cell.
Expected: "Klient LLM gotowy! Model: ..." (or "LLM niedostępny" if no LLM running — both are OK)

- [ ] **Step 4: Commit**

```bash
git add RAG_warsztat.ipynb
git commit -m "RAG notebook: update intro with FC bridge and 3-method roadmap"
```

---

### Task 2: Section 3 — Most z Function Calling (substring search limits)

**Files:**
- Modify: `RAG_warsztat.ipynb` (insert new cells after LLM connection)

This section brings the `search_presidents()` function from FC and demonstrates its failures. Students already saw this function — now we show WHY it fails.

- [ ] **Step 1: Add markdown cell — section header**

```markdown
## 3. Problem: substring search nie rozumie języka

W Function Calling zbudowaliśmy `search_presidents()` — wyszukiwarkę po słowach kluczowych.
Zobaczmy na czym polega jej ograniczenie:
```

- [ ] **Step 2: Add code cell — load presidents + substring search**

Copy `load_presidents()` and `search_presidents()` from FC notebook. These are needed locally because RAG notebook shouldn't depend on FC notebook being open.

```python
from pathlib import Path

def load_presidents():
    """Ładuje dane o prezydentach z pliku prezydenci_polski.md"""
    md_path = Path("prezydenci_polski.md")
    if not md_path.exists():
        return []
    text = md_path.read_text(encoding="utf-8")
    presidents = []
    current = {}
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
    """Prosta wyszukiwarka po słowach kluczowych."""
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
                if key != 'imię':
                    lines.append(f"  - {key}: {val}")
            wyniki.append("\n".join(lines))
    if wyniki:
        return "\n\n".join(wyniki)
    return f"Nie znaleziono wyników dla zapytania: {query}"

print(f"Załadowano {len(PREZYDENCI)} prezydentów z prezydenci_polski.md")
```

- [ ] **Step 3: Add code cell — demonstrate substring failures**

Use the test queries from `_rag_smart_search_notes.py`:

```python
# Zapytania na których substring search ZAWODZI:
test_queries = [
    ("Nobel",                  "odmiana: 'Nobel' ≠ 'Nobla' w tekście"),
    ("kto zbierał monety",    "semantyczne — brak takiej frazy w tekście"),
    ("najdłużej rządził",     "synonim: 'rządził' ≠ 'kadencja'"),
    ("związkowiec",           "w tekście 'NSZZ Solidarność', nie 'związkowiec'"),
    ("zginął tragicznie",     "w tekście 'katastrofa smoleńska', nie 'zginął'"),
    ("Kwaśniewski",           "to akurat znajdzie — bo szuka po nazwisku"),
]

print("Substring search — test:\n")
for query, why in test_queries:
    result = search_presidents(query)
    found = "Nie znaleziono" not in result
    status = "ZNALAZŁ ✓" if found else "NIE ZNALAZŁ ✗"
    print(f"  \"{query}\"")
    print(f"    → {status}  ({why})")
    if found:
        first_line = result.split("\n")[0]
        print(f"    → {first_line}")
    print()

print("Substring search działa tylko gdy pytamy DOKŁADNYMI słowami z tekstu.")
print("A gdyby LLM mógł sam przeczytać cały dokument...?")
```

- [ ] **Step 4: Commit**

```bash
git add RAG_warsztat.ipynb
git commit -m "RAG notebook: add section 3 — substring search limitations from FC"
```

---

### Task 3: Section 4 — Context Stuffing (stepping stone to RAG)

**Files:**
- Modify: `RAG_warsztat.ipynb` (insert cells after substring demo)

Context stuffing = send entire document to LLM in the prompt. Works perfectly for small docs (prezydenci_polski.md is ~4K chars). But doesn't scale. This is the "aha" moment: students see it works, then understand WHY RAG exists.

- [ ] **Step 1: Add markdown cell — section header**

```markdown
## 4. Context stuffing — wrzuć cały dokument do LLM-a

Najprostsze rozwiązanie: zamiast szukać, **dajmy LLM-owi cały tekst** i niech sam znajdzie odpowiedź.

```
Użytkownik: "Kto zbierał monety?"
    ↓
Prompt: "Oto dane o prezydentach: [CAŁY TEKST]. Odpowiedz na pytanie: ..."
    ↓
LLM: "Aleksander Kwaśniewski — w tekście jest mowa o numizmatyce"
```

To działa! Ale jest haczyk — **co jeśli dokument ma 1000 stron?** LLM ma limit tokenów (context window).
Zobaczmy jak to wygląda na naszym małym zbiorze danych:
```

- [ ] **Step 2: Add code cell — context stuffing implementation**

```python
# === Context stuffing — cały dokument w prompcie ===

_PRESIDENTS_RAW = Path("prezydenci_polski.md").read_text(encoding="utf-8")

def context_stuffing_search(query: str) -> str:
    """Wysyła CAŁY dokument do LLM-a i pyta o odpowiedź."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content":
             "Odpowiadasz WYŁĄCZNIE na podstawie podanych danych o prezydentach. "
             "Jeśli danych brak — powiedz że nie wiesz. Nie wymyślaj. Odpowiadaj po polsku."},
            {"role": "user", "content":
             f"Dane o prezydentach:\n{_PRESIDENTS_RAW}\n\nPytanie: {query}"}
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content

# Przetestujmy na tych samych zapytaniach co substring:
if client:
    print(f"Context stuffing — cały dokument ({len(_PRESIDENTS_RAW)} znaków) → LLM\n")
    for query, why in test_queries:
        print(f"  \"{query}\"")
        answer = context_stuffing_search(query)
        print(f"    → {answer[:150]}")
        print()
    print(f"Działa! Ale nasz dokument ma tylko {len(_PRESIDENTS_RAW)} znaków.")
    print("Co jeśli mamy 100 000 znaków? Albo milion? LLM tego nie pomieści.")
else:
    print("LLM niedostępny — przejdź dalej do embeddingów.")
```

- [ ] **Step 3: Add exercise 1 markdown + code cell**

```markdown
### Ćwiczenie 1: Porównaj substring vs context stuffing

Uruchom OBA podejścia na własnym pytaniu i porównaj wyniki.
W którym przypadku odpowiedź jest lepsza?
```

```python
# === Ćwiczenie 1 ===

MOJE_PYTANIE = "kto miał związek z wojskiem?"  # ← ZMIEŃ NA SWOJE

### Proszę poniżej uzupełnić kod ### (≈ 4 linijki)
# Wskazówka: wywołaj search_presidents() i context_stuffing_search() z MOJE_PYTANIE

wynik_substring = search_presidents(MOJE_PYTANIE)
wynik_stuffing = context_stuffing_search(MOJE_PYTANIE) if client else "(LLM niedostępny)"

### Koniec uzupełniania kodu ###

print(f"Pytanie: \"{MOJE_PYTANIE}\"\n")
print(f"Substring search:\n  {wynik_substring[:200]}\n")
print(f"Context stuffing:\n  {wynik_stuffing[:200]}")
```

- [ ] **Step 4: Add comparison table markdown**

```markdown
## 5. Trzy podejścia — porównanie

| | Substring search | Context stuffing | RAG (embeddingi) |
|---|---|---|---|
| **Rozumie synonimy?** | ✗ | ✓ | ✓ |
| **Rozumie kontekst?** | ✗ | ✓ | ✓ |
| **Skaluje się?** | ✓ (szybkie) | ✗ (limit tokenów) | ✓ |
| **Wymaga LLM-a?** | ✗ | ✓ | ✓ (do embeddingów nie) |
| **Koszt** | ~0 | Wysoki (cały dokument) | Niski (tylko top-K) |

Context stuffing to świetne rozwiązanie na małe zbiory danych (do ~10 stron).
Ale gdy mamy **setki lub tysiące** dokumentów — potrzebujemy RAG.

Teraz zbudujemy RAG krok po kroku!
```

- [ ] **Step 5: Commit**

```bash
git add RAG_warsztat.ipynb
git commit -m "RAG notebook: add context stuffing section with exercise + comparison table"
```

---

### Task 4: Sections 6–8 — Chunking, Embeddings, Cosine Similarity

**Files:**
- Modify: `RAG_warsztat.ipynb` (reuse/adapt existing cells from current notebook)

These sections are mostly intact from the current notebook. The key change: better introduction text referencing the 3-method table, and the data source is `prezydenci_polski_biografie.txt` (738-line biographies, not the short 55-line `.txt` or the structured `.md`).

- [ ] **Step 1: Add section 6 — document loading + chunking**

Markdown header:

```markdown
## 6. Przygotowanie dokumentów — chunking

Mamy plik `prezydenci_polski_biografie.txt` z rozbudowanymi biografiami prezydentów Polski.
Zanim zbudujemy RAG, musimy podzielić tekst na **fragmenty (chunks)**.

### Dlaczego dzielimy na fragmenty?
- Modele embeddingowe najlepiej działają na krótszych tekstach (1–3 akapity)
- Chcemy wyszukać **konkretny** fragment, nie cały dokument
- To samo robią profesjonalne systemy RAG (LangChain, LlamaIndex, itd.)
```

Then the code cells for loading, chunking, and Exercise 2 (chunk size experiment) — **keep the existing `chunk_text()` function and exercise exactly as they are now** (cells currently at positions 10–13 in the notebook). No changes needed to the code.

- [ ] **Step 2: Add section 7 — embeddings**

Keep existing markdown explanation about sentence embeddings + code for loading model and encoding chunks. **No changes to code** — cells currently at positions 14–16.

- [ ] **Step 3: Add section 8 — Exercise 3 (cosine similarity)**

Keep existing Exercise 2 (renumbered to Exercise 3). The manual cosine similarity implementation with `np.dot` and `np.linalg.norm`. **No changes to code** — cells currently at positions 17–18.

- [ ] **Step 4: Verify section flow and renumber exercises**

Current exercise numbering:
- Exercise 1 (chunk size) → becomes **Exercise 2**
- Exercise 2 (cosine similarity) → becomes **Exercise 3**

Update markdown headers accordingly.

- [ ] **Step 5: Commit**

```bash
git add RAG_warsztat.ipynb
git commit -m "RAG notebook: reorganize chunking/embeddings/cosine sections, renumber exercises"
```

---

### Task 5: Sections 9–13 — Semantic Search, RAG, Comparison, Exercise 4

**Files:**
- Modify: `RAG_warsztat.ipynb` (reuse/adapt existing cells)

These sections are mostly intact. Key changes:
- Better intro text connecting to the 3-method comparison
- Exercise 3 (custom prompt) renumbered to Exercise 4
- The "BEZ RAG vs Z RAG" comparison stays as-is

- [ ] **Step 1: Keep sections 9–11 (semantic search, RAG prompt, comparison)**

The existing `search()`, `build_rag_prompt()` functions and the comparison code work well. **No code changes.** Just verify the markdown headers flow naturally from the new context stuffing section.

- [ ] **Step 2: Renumber Exercise 3 → Exercise 4 (custom RAG prompt)**

Update the markdown header from "Ćwiczenie 3" to "Ćwiczenie 4" in the prompt modification exercise.

- [ ] **Step 3: Verify the "Zadaj własne pytanie" interactive section**

Keep as-is — students type their own question and see RAG in action.

- [ ] **Step 4: Commit**

```bash
git add RAG_warsztat.ipynb
git commit -m "RAG notebook: renumber exercises, verify search/RAG/comparison sections"
```

---

### Task 6: Sections 14–15 — Visualization + Summary

**Files:**
- Modify: `RAG_warsztat.ipynb` (update summary)

- [ ] **Step 1: Keep t-SNE visualization as-is**

The visualization code works well and doesn't need changes.

- [ ] **Step 2: Update summary markdown**

Replace the current summary with one that references the full journey:

```markdown
## Podsumowanie

### Co zrobiliśmy?

1. **Substring search** — zobaczyliśmy że wyszukiwanie po słowach kluczowych nie rozumie języka
2. **Context stuffing** — wrzuciliśmy cały dokument do LLM-a → działa, ale nie skaluje się
3. **Chunking** — podzieliliśmy tekst na fragmenty
4. **Embeddingi** — zamieniliśmy fragmenty na wektory liczbowe (384 wymiary)
5. **Wyszukiwanie semantyczne** — znaleźliśmy fragmenty najbliższe pytaniu (cosine similarity)
6. **RAG** — połączyliśmy wyszukiwanie z LLM-em → poprawne odpowiedzi!

### Trzy podejścia — kiedy co?

| | Substring | Context stuffing | RAG |
|---|---|---|---|
| **Kiedy?** | Mały, strukturalny zbiór | < 10 stron tekstu | Duże zbiory dokumentów |
| **Przykład** | `search_presidents()` z FC | Cały .md do prompta | Embeddingi + top-K + LLM |

### RAG w produkcji

W prawdziwych zastosowaniach zamiast naszego prostego kodu używa się:
- **Wektorowych baz danych** (Pinecone, Weaviate, ChromaDB, FAISS) zamiast numpy
- **Frameworków** (LangChain, LlamaIndex) do orkiestracji
- **Lepszych strategii chunkingu** (recursive, semantic chunking)
- **Re-rankingu** (dodatkowy model który ponownie ocenia trafność)
- **Hybrydowego wyszukiwania** (embeddingi + klasyczny BM25)
- **Multi-query RAG** (LLM generuje kilka wariantów pytania → lepsze pokrycie)

Ale **zasada jest dokładnie taka sama** jak to co zrobiliśmy na tych zajęciach!

### Kiedy RAG, a kiedy fine-tuning?

| | RAG | Fine-tuning (SFT) |
|---|---|---|
| **Kiedy?** | Dane się zmieniają, trzeba cytować źródła | Chcemy zmienić "styl" lub "wiedzę" modelu |
| **Koszt** | Tani (tylko inference) | Drogi (trening na GPU) |
| **Aktualność** | Zawsze aktualne (aktualizujesz dokumenty) | Wymaga ponownego treningu |
| **Halucynacje** | Mniejsze (model ma kontekst) | Możliwe (model "pamięta" błędnie) |
```

- [ ] **Step 3: Commit**

```bash
git add RAG_warsztat.ipynb
git commit -m "RAG notebook: update summary with 3-method journey recap"
```

---

### Task 7: Full Integration Test

**Files:**
- Read: `RAG_warsztat.ipynb` (run all cells)

- [ ] **Step 1: Restart kernel and run all cells**

Open the notebook in Jupyter, restart kernel, run all cells top to bottom.

Expected:
- Cells 0–7: config loads, LLM connects (or gracefully reports unavailable)
- Section 3: substring search shows failures on semantic queries
- Section 4: context stuffing answers correctly (if LLM available)
- Sections 6–8: chunking, embeddings load, cosine similarity computes
- Sections 9–11: semantic search works, RAG prompt gives correct answers
- Section 14: t-SNE plot renders

- [ ] **Step 2: Verify exercise cells have fill-in markers**

Check that all 4 exercises have:
- `### Proszę poniżej uzupełnić kod ###` marker
- `### Koniec uzupełniania kodu ###` marker
- Solution code filled in (students see working example, but can modify)

- [ ] **Step 3: Verify no `OLLAMA_URL` or stale references remain**

```bash
python3 -c "
import json
with open('RAG_warsztat.ipynb') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    for bad in ['OLLAMA_URL', 'INSTRUCTOR_SERVER', 'detect_ollama', 'detect_lmstudio', 'gdown']:
        if bad in src:
            print(f'Cell {i}: found stale reference: {bad}')
print('Scan complete')
"
```

Expected: "Scan complete" with no findings.

- [ ] **Step 4: Clean up `_rag_smart_search_notes.py`**

This file is no longer needed — its content has been incorporated into the notebook. Delete it.

```bash
git rm _rag_smart_search_notes.py
```

- [ ] **Step 5: Final commit**

```bash
git add RAG_warsztat.ipynb
git commit -m "RAG notebook: integration test passed, remove archived notes file"
```

---

## Exercise Summary

| # | Tytuł | Typ | Sekcja | Umiejętność |
|---|---|---|---|---|
| **1** | Porównaj substring vs context stuffing | Fill-in ~4 linijki | 4 | Rozumienie różnic metod |
| **2** | Eksperymentuj z rozmiarem chunków | Fill-in ~2 linijki | 6 | Wpływ chunk_size na wyniki |
| **3** | Policz cosine similarity ręcznie | Fill-in ~3 linijki | 8 | Implementacja miary podobieństwa |
| **4** | Zmodyfikuj prompt RAG | Fill-in ~5 linijek | 12 | Prompt engineering dla RAG |
