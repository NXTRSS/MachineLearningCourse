# Konwencje edycji notebooków — wewnętrzne notatki dla Claude

> Plik gitignorowany. Przekazuj go do kolejnych sesji edytowania notebooków.
> Uzupełnienie publicznego CLAUDE.md — tamten ma reguły collapse, tu są wszystkie szczegóły.

---

## 1. Struktura bloków Podpowiedź / Rozwiązanie

### Nagłówki — obowiązkowe emoji i kolory

```markdown
###### <span style="color: #c17f24;">💡 Podpowiedź</span> <span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>

###### <span style="color: #5a8a6a;">✅ Rozwiązanie</span> <span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>
```

- 💡 = Podpowiedź (złoty kolor `#c17f24`)
- ✅ = Rozwiązanie (zielony kolor `#5a8a6a`)
- Podtytuł `(kliknij aby rozwinąć)` — szary, mniejszy

### Wymagana struktura komórek (jedna komórka = jeden element)

```
[md, collapsed=True]  ###### 💡 Podpowiedź ...         ← TYLKO nagłówek, nic więcej
[md]                  treść podpowiedzi                ← osobna komórka
[md, collapsed=True]  ###### ✅ Rozwiązanie ...         ← TYLKO nagłówek
[md/code]             treść rozwiązania                ← osobna komórka (może być code)
[md]                  ###### (następne zadanie/separator) ← ZAMYKAJĄCY nagłówek — OBOWIĄZKOWY (UWAGA: spacja po ######!)
```

**Kluczowa zasada:** nagłówek `######` i jego treść MUSZĄ być w osobnych komórkach.
JupyterLab collapse chowa komórki *po* nagłówkowej — nie tekst wewnątrz niej.

### Zamykający nagłówek

Po ostatniej komórce bloku rozwiązania MUSI być komórka `###### ` zatrzymująca collapse.
Może być następne zadanie, nowa sekcja, albo pusty separator `###### `.
Bez tego — collapse "wchłonie" komórki, które mają być widoczne.

> ### ⛔⛔⛔ KRYTYCZNE — SPACJA PO ###### ⛔⛔⛔
>
> **KAŻDY separator `######` MUSI MIEĆ SPACJĘ NA KOŃCU!**
>
> ✅ Poprawnie: `###### ` (sześć hashy + SPACJA)
> ❌ Źle: `######` (bez spacji — COLLAPSE NIE ZADZIAŁA!)
>
> Bez spacji JupyterLab NIE rozpoznaje tego jako nagłówek h6
> i collapse POCHŁONIE kolejne komórki, które mają być widoczne.
>
> Dotyczy ABSOLUTNIE KAŻDEGO użycia `######` w notebookach:
> separatorów, nagłówków Podpowiedź/Rozwiązanie, Spodziewany wynik.
>
> **NIGDY nie pisz `######` bez spacji. ZAWSZE `###### `.**

### Blok 📊 Spodziewany wynik

Opcjonalny blok pokazujący studentom oczekiwany **duży** output (np. architektura sieci, tabela parametrów).
Zwinięty domyślnie — student może rozwinąć po wykonaniu ćwiczenia, żeby porównać wynik.

```markdown
###### <span style="color: #4a7090;">📊 Spodziewany wynik</span> <span style="color: #999; font-weight: normal; font-size: 0.85em;">(kliknij aby rozwinąć)</span>
```

- 📊 = Spodziewany wynik (niebieski kolor `#4a7090`)
- Treść: markdown z obrazkiem, tabelą, lub `<span style="font-family:Monospace">` z wartościami

**Struktura komórek:**

```
[code]                model.summary()                      ← komórka generująca output
[md, collapsed=True]  ###### 📊 Spodziewany wynik ...      ← nagłówek zwijający
[md]                  <div> treść (screenshot / wartości)   ← chowana przy collapse
[md]                  ###### (zamykający nagłówek)          ← OBOWIĄZKOWY
[code]                plot_model(...)                       ← następna komórka — musi być POZA collapse
```

**UWAGA — zamykający nagłówek po każdym bloku Spodziewany wynik!**
Bez niego collapse wciąga kolejne komórki (np. `plot_model`, kolejne ćwiczenie).
Jeśli po treści Spodziewanego wyniku od razu jest nowy blok Spodziewany wynik lub sekcja `#####`,
zamykający nagłówek nie jest potrzebny (sam `######` lub `#####` go zatrzymuje).
Ale jeśli po treści jest komórka code — zamykający `###### ` jest OBOWIĄZKOWY.

**Kiedy stosować (zwinięty blok Spodziewany wynik):**
- Po `model.summary()` — pełna tabela architektury
- Po `plot_model()` — screenshot grafu sieci
- Inne duże outputy (wieloliniowe tabele, długie listy parametrów)

**Kiedy NIE stosować (zamiast tego: widoczny Monospace):**
- Krótkie wyniki: shape `(21612, 21)`, RMSE, accuracy, pojedyncze liczby
- Jedno-/dwuliniowe outputy
- Ploty/wykresy — student i tak je generuje sam

**Wzorzec widocznego Monospace (krótkie wyniki):**
```markdown
###### <span style="font-family:Monospace">(21612, 21)</span>
```
Zawsze widoczny, nie zwijany. Może jednocześnie pełnić rolę zamykającego nagłówka `######`.

### Stuby nie mogą blokować Run All

Notebook musi przejść **Restart Kernel and Run All** bez zatrzymywania się —
nawet gdy student nie wypełnił żadnego stubu.

Komórki z `...` (Ellipsis) lub pustymi miejscami do uzupełnienia muszą być opakowane
w `try/except`, żeby przy Run All wyświetliły przyjazny komunikat zamiast rzucać wyjątek.

**Wzorzec:**
```python
try:
    print(f'TF("sword", text1) = {tf("sword", corpus[0]):.4f}')
except (TypeError, NameError):
    print('⬆️ Uzupełnij ... w funkcjach powyżej')
```

Dotyczy też komórek, które celowo wywołują błąd (np. `most_similar('witcher')` na zbyt małym modelu):
```python
try:
    model.wv.most_similar('witcher', topn=5)
except KeyError as e:
    print(f'KeyError: {e}')
```

**Zasada:** `Restart Kernel and Run All` → zero czerwonych tracebacków, zero przerwań.

### Reprodukowalność — `reset_seed(SEED)` na początku każdej "stochastycznej" komórki

Studenci często wykonują komórki **nie po kolei** lub **kilka razy** — żeby wyniki były powtarzalne, każda komórka która robi cokolwiek losowego (inicjalizacja wag modelu, trening z dropoutem, shuffle batchy, sampling z rozkładu) musi sama zresetować wszystkie źródła losowości.

W `utils.py` jest helper:

```python
def reset_seed(seed=42):
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as _tf
        _tf.random.set_seed(seed)
    except ImportError:
        pass
```

W notebooku — jedna stała `SEED = 42` na samej górze (cell z importami) plus `reset_seed(SEED)` na początku KAŻDEJ komórki która coś losuje:

```python
# komórka tworząca model (random init wag)
reset_seed(SEED)
model = LanguageModel()

# komórka treningu (dropout, shuffle)
reset_seed(SEED)
history = model.fit(...)

# komórka samplingu (np.random.choice)
reset_seed(SEED)
nazwa = sample_model("%")
```

**Test reprodukowalności:** Restart Kernel and Run All → wszystkie outputy identyczne. Run All drugi raz → wciąż identyczne. Wykonanie pojedynczej komórki dwa razy → wynik się nie zmienia.

**Skrypty hyperparameter search** (np. `lm_search_bayesian.py`) — używają tego samego seeda i resetują przed każdym trial'em żeby search był deterministyczny:

```python
def objective(trial):
    reset_seed(SEED)  # żeby każdy trial miał te same warunki początkowe
    ...
```

Sampler Optuna też dostaje seed: `TPESampler(seed=SEED)`.

### Puste linie wokół linii do uzupełnienia

Każda linia kodu którą student ma uzupełnić (z `...` i komentarzem `# Tutaj wpisz swój kod`)
**MUSI być poprzedzona pustą linią i mieć pustą linię za sobą**. Dotyczy to nawet pojedynczej linii.

Cel — wizualne wyróżnienie w komórce; student od razu widzi gdzie ma kursor postawić.

**Źle:**
```python
try:
    city_list = ...  # Tutaj wpisz swój kod
    print(city_list[:10])
except TypeError:
    print('Proszę uzupełnić rozwiązanie powyżej')
```

**Dobrze:**
```python
try:

    city_list = ...  # Tutaj wpisz swój kod

    print(city_list[:10])
except TypeError:
    print('Proszę uzupełnić rozwiązanie powyżej')
```

Przy wielu liniach do uzupełnienia obok siebie — pusta linia przed pierwszą i po ostatniej
(nie trzeba rozdzielać każdej z osobna jeśli stanowią logiczną grupę).

### Zasady pisania treści Podpowiedzi

**Podpowiedź ma odblokować myślenie — nie zastąpić rozwiązania.**

| Sytuacja | Jak postąpić |
|----------|-------------|
| Ćwiczenie trywialne (1 linia, oczywiste API) | Nie dodawaj podpowiedzi wcale |
| Ćwiczenie proste, ale z nieoczywistym gotcha (np. `np.zeros((dim,1))`) | Podpowiedź opisująca pułapkę, bez gotowego kodu |
| Ćwiczenie średnie — trzeba przetłumaczyć wzór na kod | Podaj wzór matematyczny lub wskaż funkcje NumPy; nie pisz gotowych wywołań |
| Ćwiczenie złożone — kilka kroków do połączenia | Opisz kolejność kroków; bez sygnatur argumentów, bez kodu |

**Konkretne zakazy:**
- Nigdy nie wstawiaj do podpowiedzi kodu, który jest (lub blisko jest) rozwiązaniem
- Nie podawaj nazw zmiennych wynikowych ani dokładnych sygnatur wywołań
- Nie pisz snippetów które student kopiuje 1:1 — to Rozwiązanie, nie Podpowiedź

**Test:** jeśli student może skopiować treść podpowiedzi i wkleić jako rozwiązanie — jest za mocna.

---

## 2. Skrypt przed_zajeciami.py

### Co robi

1. Czyści outputy i `execution_count` ze wszystkich komórek code
2. Ustawia `jp-MarkdownHeadingCollapsed: True` na nagłówkach `######` zawierających słowa kluczowe

### Logika zwijania — WAŻNE

```python
src = "".join(cell.get("source", []))
first_line = src.lstrip().split("\n")[0]
is_h6 = first_line.startswith("######")
has_keyword = "Rozwiązanie" in first_line or "Podpowiedź" in first_line or "Spodziewany wynik" in first_line
if is_h6 and has_keyword:
    cell.setdefault("metadata", {})["jp-MarkdownHeadingCollapsed"] = True
```

Skrypt patrzy TYLKO na **pierwszą linię** i TYLKO na nagłówki `######`.
Nigdy nie zwija komórek, które zawierają te słowa w środku treści (np. intro sekcji opisujące legendę).

### Obsługuje obie wersje — z emoji i bez

Działa zarówno dla `###### Rozwiązanie` jak i `###### ✅ Rozwiązanie` —
bo sprawdza czy słowo kluczowe jest gdziekolwiek w pierwszej linii.

### Lista notebooków w skrypcie

```python
NOTEBOOKI = [
    "Regresja_liniowa_rozszerzona.ipynb",
    "Regresja_liniowa.ipynb",
    "Regresja_logistyczna.ipynb",
    "Reprezentacja_tekstu.ipynb",
    "Sieć_Neuronowa.ipynb",
    "Model_Języka_polskie_nazwy_miast.ipynb",
]
```

Przy dodawaniu nowego notebooka z blokami Podpowiedź/Rozwiązanie — dodaj go tu.

---

## 3. Co było naprawione w Regresja_liniowa.ipynb (historia zmian)

Poniżej lista konkretnych bugów i poprawek wprowadzonych przed/podczas sesji z Claude.
Przy edycji innych notebooków sprawdź czy nie mają tych samych problemów.

### Błędy techniczne (kod przestał działać z nowszymi bibliotekami)

| Problem | Stare | Nowe |
|---------|-------|------|
| `sns.distplot()` — usunięte w seaborn 0.12 | `sns.distplot(df['price'])` | `sns.histplot(df['price'], kde=True)` |
| `df.corr()` z pandas ≥ 2.0 — `numeric_only` wymagane | `df.corr()` | `df.select_dtypes(include='number').corr()` |
| `plt.legend()` przed `plt.plot()` — brak legendy | legend call przed danymi | przestawione w odpowiedniej kolejności |
| `LinearModel(None, None)` — crash przy None | brak guardu | dodano `if a0 is None` + przywrócono zmienne `a0`, `a1` |

### Wyczyszczenie wyciekłych rozwiązań

W komórkach 22 i 93 (komórki studenckie) znajdowały się gotowe rozwiązania zamiast placeholder-ów.
Zastąpione przez `# Tutaj wpisz swój kod`.

**Wzorzec dla komórek studenckich:**
```python
# Tutaj wpisz swój kod
```
Nigdy `None` ani puste komórki — to myli studentów.

**Wyjątek — `pass` wewnątrz bloków kodu:** sam komentarz w ciele `for`/`if`/funkcji to błąd składni Pythona. W takich miejscach użyj:
```python
pass  # Tutaj wpisz swój kod
```

### Dodane podpowiedzi i rozwiązania (d4c4007, 5671734)

9 ćwiczeń otrzymało bloki Podpowiedź + Rozwiązanie w nowym formacie (patrz sekcja 1).
Wcześniej notebook nie miał żadnych podpowiedzi.

### Tekst uzasadnienia klasy LinearModel (bfcfb0b, 206aeb0, e6568ee)

Przepisano tekst wyjaśniający po co tworzyć klasę (zamiast funkcji) — był zbyt suchy.
Dodano wykres scatter z linią modelu przy przejściu do wielowymiarowego modelu.
Callout "Dlaczego klasa?" wystylizowano jako niebieski info box:
```markdown
<div style="background:#e8f4f8; border-left:4px solid #2196F3; padding:12px; border-radius:4px;">
...
</div>
```

### Kompatybilność środowisk (174a9f2)

Dodano `ensure_package()` — komórka setup na początku notebooka, która:
- na Dockerze/uv: pomija instalację (paczki już są)
- na Colab: instaluje przez pip

Usunięto wszystkie hardcoded ścieżki `/content/` (Colab-specific).
Usunięto zakomentowane `!conda install` i `!pip install` rozrzucone po notebooku.

### Pobieranie danych (7a470a3)

Dane `data_houses.csv` pobierane automatycznie z GitHub przez `utils.py` (fallback z Google Drive).
Nie trzeba ręcznie umieszczać pliku CSV w folderze.

### Emoji — używaj oszczędnie

Dozwolone tylko dwa: `💡` (Podpowiedź) i `✅` (Rozwiązanie) — wyłącznie w nagłówkach tych bloków.
Nie dodawaj emoji do tytułów sekcji, komentarzy kodu, `set_title()`, print(), ani komunikatów.
Wygląda infantylnie i rozprasza.

---

## 3b. Co było naprawione w Regresja_logistyczna.ipynb (historia zmian)

### Wyczyszczenie wyciekłych rozwiązań (sesja 2026-04-27)

Wyciekłe rozwiązania znajdowały się w **komórkach studenckich** (nie pod nagłówkami Rozwiązanie):

| Komórka (oryg.) | Problem | Poprawka |
|----------------|---------|----------|
| 24 | `random.shuffle(all_data_processed)` widoczna odpowiedź | `# Tutaj wpisz swój kod` |
| 37 | Pełny kod podziału train/test | Stub z `split_ratio` i `split_idx`, `# Tutaj wpisz swój kod` |
| 62 | `s = 1 / (1 + np.exp(-z))` między markerami w sigmoid | `# Tutaj wpisz swój kod` |
| 70 | `w = np.zeros(dim, 1)` (BUG: brakuje nawiasów!) + `b = 0` | `# Tutaj wpisz swój kod` |
| 78 | Pełne forward + backward propagation | `# Tutaj wpisz swój kod` (sekcje) |
| 85 | Pełna pętla gradient descent | `# Tutaj wpisz swój kod` (sekcje) |
| 92 | Pełne predict (sigmoid + threshold) | `# Tutaj wpisz swój kod` (sekcje) |
| 99 | Pełny model (bug: `X, Y` zamiast `X_train, Y_train`) | `# Tutaj wpisz swój kod` (sekcje z komentarzami) |

### Dodane bloki Podpowiedź (sesja 2026-04-27)

8 ćwiczeń otrzymało bloki `💡 Podpowiedź` wstawione przed istniejącymi `✅ Rozwiązanie`.
Każda podpowiedź zawiera wzory matematyczne w LaTeX i snippety kodu.

Ćwiczenia z podpowiedziami:
1. `random.seed` + `shuffle`
2. Podział train/test przez slice
3. Funkcja sigmoid $\sigma(z) = 1/(1+e^{-z})$
4. `initialize_with_zeros` — uwaga na podwójne nawiasy `np.zeros((dim, 1))`
5. `propagate` — forward + backward prop z gradientami
6. `optimize` — pętla gradient descent
7. `predict` — sigmoid + threshold 0.5
8. `model` — złożenie wszystkich funkcji

### Zmiany techniczne (sesja 2026-05-02)

**Stała RESAMPLING** — dodana w komórce setup (zaraz po `IMG_SIZE = 64`):
```python
RESAMPLING = Image.LANCZOS  # filtr próbkowania: LANCZOS = najwyższa jakość przy zmniejszaniu obrazu
```
Wszędzie w notebooku zamiast `Image.LANCZOS` używamy `RESAMPLING`.
Zasada ogólna: stałe konfiguracyjne (rozmiary, filtry) definiujemy raz na górze, nie rozrzucamy po komórkach.

**SelectFilesButton** — przepisany w `utils.py` z tkinter na `widgets.FileUpload`:
```python
class SelectFilesButton(widgets.FileUpload):
    def __init__(self):
        super().__init__(accept='image/*', multiple=True,
                         description='Wybierz zdjęcia', button_style='warning',
                         layout=widgets.Layout(width='auto'))
```
Stara wersja oparta na tkinter nie działała w środowiskach bez GUI (Docker, JupyterHub, VS Code).
W ipywidgets 8.x: `my_button.value` to tuple słowników `{name, type, size, last_modified, content: memoryview}`;
bajty obrazka: `file_info['content'].tobytes()`.

**Komórka predykcji z pliku (cell 115)** — używa nowego API ipywidgets 8.x, wyświetla
prawdopodobieństwo i kolor tytułu (zielony = kot, czerwony = nie-kot).

**Komórka predykcji z URL (cell 116)** — uproszczona do jednego aktywnego URL
z 10 zakomentowanymi przykładami (4 TP, 4 TN, 1 FP, 1 FN) z opisem predykcji modelu.

**Uwaga — ipywidgets version mismatch** (środowisko `ml_wsb`):
Kernel `ml_wsb` ma ipywidgets 7.6.5, rozszerzenie JupyterLab pochodzi z ipywidgets 8.x
(`jupyterlab_widgets 3.0.16`). Objaw: "Click to show javascript error" / `^1.5.0 not registered, 2.0.0 is`.
Rozwiązanie docelowe: `conda install -n ml_wsb ipywidgets>=8` lub `pip install ipywidgets>=8`.
Na razie SelectFilesButton działa mimo błędu JS (widget renderuje się poprawnie w JupyterLab 4).

---

---

## 3c. Konwencje pobierania obrazków w notebookach

### Zawsze używaj User-Agent

Wiele serwerów (w szczególności Wikimedia) blokuje zwykłe żądania bez nagłówka User-Agent (HTTP 403).
Standardowy wzorzec dla `urllib`:

```python
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
image = Image.open(io.BytesIO(urllib.request.urlopen(req).read())).convert('RGB')
```

### Hierarchia źródeł — od najbardziej do najmniej niezawodnych

| Źródło | Uwagi |
|--------|-------|
| `raw.githubusercontent.com` | Najlepsze — bez auth, bez rate limit, bez ograniczeń rozmiaru |
| `upload.wikimedia.org` (oryginały, nie `/thumb/`) | Dobre, ale wymaga User-Agent; pod dużym obciążeniem zwraca 429 |
| Google Storage (`storage.googleapis.com`) | Zwykle publiczne, ale zależy od bucketu |
| HuggingFace datasets | Wymaga auth dla niektórych zbiorów |
| Bezpośrednie linki z Google Images / stron komercyjnych | Niestabilne — często zmieniają URL |

### Dobre repozytoria obrazków do demonstracji

- **`EliSchwartz/imagenet-sample-images`** na GitHub — 1000 przykładowych obrazków ImageNet, każdy w osobnym pliku JPEG, nazwany wg klasy (np. `n02123597_Siamese_cat.JPEG`)
- **`tensorflow/models`** — `research/object_detection/test_images/image1.jpg` (beagle), `image2.jpg` (plaża)
- **`fastai/fastai`** — `nbs/images/cat.jpg` — dobry TP dla modelu catvsnotcat

### Weryfikacja URL przed commitem

Przed wstawieniem URL do notebooka: sprawdź każdy z nich w środowisku notebooka (lub przez curl/Python) — upewnij się że:
1. Serwer zwraca 200 (nie 403, 404, 429)
2. Obraz otwiera się przez PIL
3. Predykcja modelu jest taka jak opisana w komentarzu

Nie wstawiaj "prawdopodobnie działających" URLi — tylko zweryfikowane.

### Thumbnails Wikimedia — pułapka

URL z `/thumb/` (np. `https://upload.wikimedia.org/wikipedia/commons/thumb/X/X/file.jpg/400px-file.jpg`)
akceptuje tylko konkretne rozmiary. Przy złym rozmiarze: HTTP 400.
Zamiast tego używaj URL do pełnego oryginału (bez `/thumb/` i bez `NNpx-` prefixu).

---

## 4. Podcasty — TYLKO w Regresja_liniowa_rozszerzona.ipynb

**Podcasty są TYLKO w tym jednym notebooku** — to materiał do samodzielnej pracy w domu przed zajęciami.
Przy edycji pozostałych notebooków (Regresja_liniowa, Regresja_logistyczna, Sieć_Neuronowa itd.) NIE dodawaj podcastów.

### Format komórki odtwarzacza

```python
from IPython.display import Audio, display, HTML
display(HTML("<b>🎙️ Odcinek N — Tytuł odcinka</b>"))
display(Audio("https://raw.githubusercontent.com/NXTRSS/MachineLearningCourse/main/podcast_audio_xtts/NN_slug.mp3"))
```

- ID komórki: `podcast-ep01` ... `podcast-ep10`
- Pliki MP3: `podcast_audio_xtts/` w repo, gałąź `main`
- URL musi wskazywać na `main` (nie `docker-uv-setup` ani inną gałąź)

### Mapowanie odcinków → sekcje

| ID komórki | Plik MP3 | Sekcja notebooka |
|------------|----------|------------------|
| podcast-ep01 | 01_intro.mp3 | ## Trzy grupy algorytmów |
| podcast-ep02 | 02_dane.mp3 | ## Ściągnięcie danych |
| podcast-ep03 | 03_po_co_eda.mp3 | ## Dlaczego wizualizujemy |
| podcast-ep04 | 04_wizualizacje.mp3 | ## Wizualizacje |
| podcast-ep05 | 05_najlepsza_linia.mp3 | ## Intuicja — najlepsza linia |
| podcast-ep06 | 06_regresja_2d.mp3 | ## Regresja 2D |
| podcast-ep07 | 07_z_2d_do_3d.mp3 | ## Z 2D do 3D |
| podcast-ep08 | 08_regresja_3d.mp3 | ## Regresja 3D |
| podcast-ep09 | 09_wielomiany.mp3 | ## Wielomiany |
| podcast-ep10 | 10_overfitting.mp3 | ## Regresja wielomianowa |

---

## 5. Konwencje cell ID

Cell ID to UUID przypisywany przez Jupyter. Przy dodawaniu nowych komórek przez skrypt:
- Używaj unikalnych ID w formacie `podcast-ep01`, `podcast-ep02` itd. dla podcastów
- Dla komórek ćwiczeń nie zmieniaj ID jeśli już istnieją (mogą być linkowane)

---

## 6. Edycja .ipynb gdy notebook jest otwarty w JupyterLab

> ### ⛔⛔⛔ KRYTYCZNE — JUPYTERLAB NADPISUJE ZMIANY ⛔⛔⛔
>
> JupyterLab trzyma notebook w pamięci i co kilka sekund zapisuje swój bufor na dysk (autosave).
> Jeśli edytujesz `.ipynb` skryptem/CLI gdy notebook jest otwarty — JupyterLab **natychmiast nadpisze**
> Twoje zmiany swoją starą wersją z pamięci. Zmiany przepadną bez ostrzeżenia.
>
> **ZAWSZE zamknij sesję notebooka przez API PRZED edycją pliku `.ipynb`!**

### Procedura bezpiecznej edycji

1. **Zamknij sesję notebooka przez Jupyter REST API:**

```bash
TOKEN=$(uv run jupyter server list 2>&1 | grep -oP 'token=\K[a-f0-9]+')
SESSION_ID=$(curl -s "http://localhost:8888/api/sessions?token=$TOKEN" \
  | python3 -c "
import sys, json
for s in json.load(sys.stdin):
    if 'NAZWA_NOTEBOOKA' in s.get('path', ''):
        print(s['id']); break
")
curl -s -X DELETE "http://localhost:8888/api/sessions/$SESSION_ID?token=$TOKEN"
```

2. **Edytuj plik** (skryptem, Pythonem, sed — cokolwiek)

3. **Użytkownik otwiera notebook ponownie** — JupyterLab ładuje świeżą wersję z dysku

### Alternatywa — wyłącz autosave w notebooku

W pierwszej komórce notebooka:
```python
%autosave 0
```
To wyłącza autosave dla tej sesji. Ale nadal: jeśli użytkownik naciśnie Ctrl+S, nadpisze plik.

### Dlaczego `File → Revert Notebook` nie wystarcza

Revert ładuje plik z dysku, ale **nie zamyka sesji**. Jeśli użytkownik nie zrevertuje
(albo JupyterLab zrobi autosave zanim zdąży) — zmiany przepadają.
Zamknięcie sesji przez API jest jedynym niezawodnym rozwiązaniem.

---

## 6b. Zapis notebooków przez skrypt

```python
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
```

Zawsze `indent=1` i `ensure_ascii=False` (polskie znaki jako UTF-8, nie `\uXXXX`).
Zawsze newline na końcu pliku.

---

## 7. Zarządzanie zależnościami — lekki Docker vs ensure_package

### Zasada: pyproject.toml i requirements.txt muszą być jak najlżejsze

Pliki `pyproject.toml` i `requirements.txt` definiują bazowy obraz Dockera.
Każda dodatkowa paczka = dłuższy build. Wrzucaj tam **tylko** zależności:
- używane w **wielu** notebookach (numpy, pandas, tensorflow, scikit-learn, matplotlib...)
- ciężkie do zainstalowania na bieżąco (tensorflow, spacy, torch...)

### ensure_package — dla rzadkich / lekkich zależności

Paczki używane w jednym lub dwóch notebookach, które instalują się szybko
(np. `tiktoken`, `instructor`, `pydantic`), instaluj przez `ensure_package()` na początku notebooka:

```python
from utils import ensure_package
ensure_package("tiktoken")
```

`ensure_package` ma fallback chain: uv → pip. Na Dockerze/uv paczka może już być;
na Colab zainstaluje się przez pip — działa wszędzie.

**Nie dodawaj** takich paczek do `pyproject.toml` / `requirements.txt`.

### pyproject.toml — aktualny stan zależności

```toml
requires-python = ">=3.9,<3.13"

"tensorflow>=2.16"        # 2.16+ ma realne wheels na Win/Mac ARM64/Linux
"spacy>=3.7"              # ma wheels dla Windows — bez restrykcji platformy
"pydot>=2.0"              # stary pydot<2.0 był sdist-only
```

Poprzednia wersja miała `tensorflow==2.15; sys_platform != 'win32'` — to było złe,
Windows nie miał tensorflow wcale. Od 2.16 wheele są wszędzie.

---

## 8. Co NIE idzie do main (pliki tylko dla prowadzącego)

- `przed_zajeciami.py` — skrypt do przygotowania notebooków przed zajęciami
- `NLP_filled.ipynb` — wersja z wypełnionymi rozwiązaniami
- `_build_fc_notebook.py` — skrypt budujący Function_Calling.ipynb
- `RAG_warsztat.ipynb` — w trakcie budowy
- `podcast_xtts.ipynb` — notebook do generowania TTS
- `monitor_server.py` — narzędzie prowadzącego
- `docs/TODO_ngrok_instructor_server.md` — notatki planistyczne

Zostają na gałęzi `docker-uv-setup`.

---

## 9. Gałęzie repo

| Gałąź | Zawartość |
|-------|-----------|
| `main` | Dla studentów — notebooki gotowe, docs, pyproject.toml |
| `docker-uv-setup` | Robocza — wszystkie pliki włącznie z instruktorskimi |
| `claude/great-chatterjee-2c4d55` | Worktree z poprzedniej sesji (można ignorować) |
