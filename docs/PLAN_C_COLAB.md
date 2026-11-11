# Plan C — Google Colab (gdy nic innego nie działa)

Google Colab to darmowe środowisko Jupyter w przeglądarce, utrzymywane przez Google. Nie wymaga żadnej instalacji — wystarczy konto Google.

Jest to **rozwiązanie awaryjne** — działa zawsze, ale ma ograniczenia (sesje się rozłączają po ~90 minutach bezczynności, wersje pakietów mogą się różnić).


## Krok 1: Otwarcie notebooka z GitHuba

1. Wejdź na: https://colab.research.google.com
2. Kliknij **File** → **Open notebook** → zakładka **GitHub**
3. Wklej `https://github.com/NXTRSS/MachineLearningCourse` i wybierz notebook (np. `Regresja_liniowa.ipynb`)


## Krok 2: (alternatywa) Wgranie notebooka z dysku

Jeśli otwieranie z GitHuba nie działa: pobierz repozytorium jako ZIP (zielony przycisk **Code** → **Download ZIP** na https://github.com/NXTRSS/MachineLearningCourse), wypakuj, a w Colabie wybierz **File** → **Upload notebook**.


## Krok 3: Instalacja pakietów

Na samym początku notebooka (przed pierwszą komórką z kodem) dodaj nową komórkę i wklej:

```python
!pip install -q "tensorflow>=2.20" "pillow>=9.4.0" "pandas>=2.2.3" "scikit-learn>=1.6.0" \
    "seaborn>=0.13.0" "plotly>=5.1.0" "pydot>=2.0" "graphviz>=0.20.3" \
    "matplotlib>=3.9.2" "ipywidgets>=8.1.2" gdown packaging
```

(Cudzysłowy są potrzebne — bez nich powłoka potraktuje `>=` jako przekierowanie do pliku.)

Uruchom tę komórkę (`Shift+Enter`). Instalacja potrwa 1-2 minuty.

> **Uwaga:** Colab może wyświetlić ostrzeżenie o restarcie sesji — kliknij **"Restart runtime"** jeśli się pojawi. Komórki instalacyjnej nie trzeba uruchamiać powtórnie — pakiety zostały już zainstalowane; po restarcie uruchom dalsze komórki od początku.


## Krok 4: Wgranie plików pomocniczych

Niektóre notebooki korzystają z pliku `utils.py`. Aby go wgrać do Colab:

### Opcja A: Ręczne wgranie

1. W panelu po lewej stronie kliknij ikonkę folderu (📁)
2. Kliknij ikonkę wgrywania (strzałka w górę)
3. Wybierz plik `utils.py` ze swojego komputera

### Opcja B: Pobranie z GitHuba w komórce

Dodaj komórkę na początku notebooka:

```python
!wget https://raw.githubusercontent.com/NXTRSS/MachineLearningCourse/main/utils.py
```


## Krok 5: Dane do zajęć

Pliki z danymi (np. `data_houses.csv`) zostaną pobrane automatycznie przez `utils.py` (z repozytorium kursu na GitHubie).

Jeśli automatyczne pobieranie nie zadziała, dane można wgrać ręcznie przez panel plików w Colabie (ikona folderu po lewej).


## Ważne informacje o Colabie

- **Sesja trwa ~12 godzin** — po tym czasie środowisko się resetuje i trzeba ponownie instalować pakiety i wgrywać pliki
- **Bezczynność ~90 minut** — Colab rozłącza sesję po ~90 minutach bez aktywności
- **Wersje pakietów** — Colab ma preinstalowane swoje wersje pakietów, które mogą się nieznacznie różnić. Komórka z `!pip install` powinna to nadpisać
- **GPU** — Colab daje darmowy dostęp do GPU. Aby go włączyć: **Runtime** → **Change runtime type** → **T4 GPU**
- **Brak lokalnego LLM-a** — na Colabie nie uruchomisz LM Studio ani Ollamy. Warsztaty z Function Calling i RAG działają wtedy tylko przez serwer prowadzącego udostępniony przez internet (tunel) — adres i hasło poda prowadzący.


## Alternatywa: MyBinder

Można też otworzyć środowisko jednym kliknięciem przez MyBinder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NXTRSS/MachineLearningCourse/HEAD)

**Uwagi:** MyBinder jest wolniejszy od Colaba, budowanie środowiska może trwać 5-10 minut, a sesje mają ograniczoną pamięć i czas.
