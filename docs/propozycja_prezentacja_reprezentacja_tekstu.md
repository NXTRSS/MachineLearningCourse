# Propozycja: osobna prezentacja "Reprezentacja tekstu i Embeddingi"

## Co jest w obecnej prezentacji (LM_kn.pdf, slajdy 22-29)

| Slajd | Tytuł | Uwagi |
|-------|-------|-------|
| 22 | Podstawy Przetwarzania Tekstu | NLTK, Brown corpus, tokenizacja — OK |
| 23 | Embeddingi w NLP - Podstawy | Czym są, typy, heatmapa — OK |
| 24 | Embeddingi w NLP - Przykład wymiarów | Meaning as vectors 2D — OK |
| 25 | TF-IDF | Wzory prosta + złożona wersja — OK |
| 26 | Reprezentacje Wektorowe Słów | Gęste wektory, zalety — OK |
| 27 | Word2Vec | Parametry, demonstrowane możliwości — OK |
| 28 | Modele Pretrenowane | w2v-google-news, GloVe — OK |
| 29 | Zastosowania Praktyczne | Przypadki użycia, najlepsze praktyki — OK |

## Co bym zachował z obecnych slajdów

Wszystkie powyższe slajdy — są dobrze przygotowane. Stanowią solidną bazę.

## Co bym dodał / zmienił

### 1. Slajd otwierający — kontekst warsztatu
- **Tytuł:** "Reprezentacja tekstu — od tekstu do wektorów"
- **Treść:** Dlaczego komputer nie rozumie tekstu? Jak zamienić słowa na liczby?
  Krótka agenda: NLTK → TF-IDF → Embeddingi → Wizualizacja
- **Dlaczego:** Obecna prezentacja zaczyna od razu od NLTK, bez motywacji

### 2. Po slajdzie NLTK — dodać slajd "Bag of Words"
- **Tytuł:** "Od tekstu do liczb — Bag of Words"
- **Treść:** Najprostsza reprezentacja: wektor zliczeń słów. Przykład z 2-3 zdaniami.
  Problem: tracimy kolejność, słowa częste dominują → motywacja do TF-IDF
- **Dlaczego:** Notebook przeskakuje od NLTK od razu do TF-IDF, brakuje BoW jako pomostu

### 3. Slajd TF-IDF — dodać praktyczny przykład
- Obecny slajd 25 ma wzory — jest OK
- Dodać drugi slajd z **minimalnym przykładem liczbowym** (2 dokumenty, 3 słowa, ręczne obliczenie)
- **Dlaczego:** Studenci lepiej zrozumieją wzory widząc konkretne liczby przed notebookiem

### 4. Po slajdzie "Embeddingi — Przykład wymiarów" — dodać "One-hot vs Dense"
- **Tytuł:** "One-hot encoding vs Embeddingi"
- **Treść:** Wizualne porównanie — one-hot (sparse, ortogonalny, brak relacji) vs embedding (dense, zachowuje semantykę)
- **Dlaczego:** Obecny slajd 26 mówi o "redukcji wymiarowości w porównaniu z one-hot" ale nie pokazuje tego kontrastowo

### 5. Slajd Word2Vec — dodać "Jak działa Word2Vec"
- **Tytuł:** "Word2Vec — intuicja działania"
- **Treść:** Idea Skip-gram / CBOW (okno kontekstu), schemat sieci.
  Nie trzeba wzorów — wystarczy diagram: "dane słowo → przewiduj sąsiednie" 
- **Dlaczego:** Obecny slajd mówi o parametrach (window=5, vector_size=100) ale nie wyjaśnia DLACZEGO te parametry mają znaczenie

### 6. Po slajdach z pretrenowanymi modelami — dodać "Porównanie: własny vs pretrenowany"
- **Tytuł:** "Mały korpus vs duży korpus"
- **Treść:** Porównanie wyników `most_similar('wine')` — Brown corpus (bzdury) vs Google News (sensowne wyniki). Screenshot z notebooka.
- **Dlaczego:** To jest kluczowy moment "aha!" w notebooku — warto go podkreślić na slajdzie

### 7. Slajd podsumowujący — "Od tekstu do NLP"
- **Tytuł:** "Co dalej? Od embeddingów do modeli językowych"
- **Treść:** Bridge slide: embeddingi → modele sekwencyjne (RNN/LSTM) → modele językowe → Transformery → GPT
  "Na następnych zajęciach: budujemy model językowy generujący polskie nazwy miast"
- **Dlaczego:** Łączy ten warsztat z następnym (LM + polskie miasta), daje studentom mapę drogową

## Proponowana kolejność slajdów w nowej prezentacji

1. **Reprezentacja tekstu — od tekstu do wektorów** (NOWY)
2. Podstawy Przetwarzania Tekstu (istniejący slajd 22)
3. **Od tekstu do liczb — Bag of Words** (NOWY)
4. TF-IDF (istniejący slajd 25)
5. **TF-IDF — przykład liczbowy** (NOWY)
6. Embeddingi w NLP - Podstawy (istniejący slajd 23)
7. **One-hot vs Dense embeddingi** (NOWY)
8. Embeddingi w NLP - Przykład wymiarów (istniejący slajd 24)
9. Reprezentacje Wektorowe Słów (istniejący slajd 26)
10. **Word2Vec — intuicja działania (Skip-gram/CBOW)** (NOWY)
11. Word2Vec (istniejący slajd 27)
12. Modele Pretrenowane (istniejący slajd 28)
13. **Mały korpus vs duży korpus — porównanie** (NOWY)
14. Zastosowania Praktyczne (istniejący slajd 29)
15. **Co dalej? Od embeddingów do modeli językowych** (NOWY — bridge do LM)

Razem: 15 slajdów (8 istniejących + 7 nowych)
