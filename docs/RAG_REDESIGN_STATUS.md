# RAG Notebook Redesign — Status (2026-05-15)

## Stan: ✅ Redesign + ćwiczenia gotowe, wymaga przetestowania na żywo

Branch: `docker-uv-setup` (pushed)

## Co się zmieniło

Notebook przebudowany z 35 → 60 cells. Nowy flow:

```
Intro (most z FC) → Substring search → Context stuffing → Porównanie 3 metod
→ Chunking → Embeddingi → Cosine similarity → Semantic search
→ RAG → Porównanie BEZ/Z RAG → Interactive → t-SNE → Podsumowanie
```

### Nowe sekcje (vs stara wersja)
- **Sekcja 3**: Substring search z FC — `search_presidents()` (oryginalna prosta wersja z `all()`) + demo failów
- **Sekcja 4**: Context stuffing — cały `prezydenci_polski.md` do LLM-a
- **Sekcja 5**: Tabela porównawcza 3 metod
- **Ćwiczenie 1**: Porównaj substring vs context stuffing (nowe)

### Zmienione
- **Intro**: Bridge z FC ("Pamiętacie search_presidents? Miał limity...")
- **Dane RAG**: `prezydenci_polski_biografie.txt` (738 linii) zamiast starego `prezydenci_polski.txt` (55 linii)
- **Ćwiczenia**: 3 → 4 (dodane porównanie substring vs stuffing)
- **Podsumowanie**: Pełna podróż substring → stuffing → RAG + tabela "kiedy co"
- **Colab cell**: Pobiera też `prezydenci_polski_biografie.txt` i `.md`

### Usunięte
- `_rag_smart_search_notes.py` — treść wchłonięta do notebooka

### Infrastruktura ćwiczeń (2026-05-15)
- **4 student-stub cells** z tagami i szablonami (`student_stub_template` w metadata)
- **Podpowiedzi** (collapsed): Ćw. 1, 3, 4 (ćw. 2 trywialne — tylko rozwiązanie)
- **Rozwiązania** (collapsed): Wszystkie 4 ćwiczenia
- **Separatory** (`###### `): Po każdym bloku podpowiedź/rozwiązanie
- **try/except guards**: Stuby nie blokują Run All
- **`przed_zajeciami.py`**: RAG notebook dodany do NOTEBOOKI — resetuje stuby, czyści outputy, zwija sekcje
- **Ściągawka**: `RAG_warsztat_ODPOWIEDZI.ipynb` generowana automatycznie do `._odp/`

## Pliki

| Plik | Status |
|---|---|
| `RAG_warsztat.ipynb` | Przebudowany (60 cells, 13 sekcji, 4 ćwiczenia, podpowiedzi/rozwiązania) |
| `prezydenci_polski_biografie.txt` | Nowy — rozbudowane biografie 7 prezydentów III RP |
| `prezydenci_polski.md` | Bez zmian — używany przez substring search i context stuffing |
| `prezydenci_polski.txt` | Bez zmian — stary plik, już nieużywany przez RAG |
| `przed_zajeciami.py` | Zaktualizowany — RAG notebook w liście NOTEBOOKI |
| `_rag_smart_search_notes.py` | Usunięty |

## Do przetestowania na żywo

- [ ] Uruchomić cały notebook z LLM-em (LM Studio / Ollama)
- [ ] Sprawdzić czy context stuffing odpowiada sensownie na test_queries
- [ ] Sprawdzić czy RAG z `prezydenci_polski_biografie.txt` (738 linii) daje lepsze chunki niż stary 55-liniowy plik
- [ ] Sprawdzić czy t-SNE wizualizacja wygląda OK z większą liczbą chunków
- [ ] Ćwiczenia — czy student-stuby mają sens, czy podpowiedzi wystarczą
- [x] `przed_zajeciami.py` przetwarza RAG notebook (4 stuby, 7 collapses, outputy)

## Znane kwestie

- `prezydenci_polski.txt` (55 linii) nadal leży w repo — nie jest używany, ale nie przeszkadza
- Nawrocki w `prezydenci_polski_biografie.txt` — dane na dzień pisania (maj 2026), mogą się zdezaktualizować
