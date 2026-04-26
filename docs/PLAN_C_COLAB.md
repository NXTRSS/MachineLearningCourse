# Plan C — Google Colab (gdy nic innego nie działa)

Google Colab to darmowe środowisko Jupyter w przeglądarce, utrzymywane przez Google. Nie wymaga żadnej instalacji — wystarczy konto Google.

Jest to **rozwiązanie awaryjne** — działa zawsze, ale ma ograniczenia (sesje się rozłączają po ~90 minutach bezczynności, wersje pakietów mogą się różnić).


## Krok 1: Pobranie plików z zajęć

1. Wejdź na: https://github.com/NXTRSS/MachineLearningCourse
2. Kliknij zielony przycisk **Code** → **Download ZIP**
3. Wypakuj archiwum na swoim komputerze


## Krok 2: Wgranie notebooka do Colab

1. Wejdź na: https://colab.research.google.com
2. Kliknij **File** → **Upload notebook**
3. Wybierz plik `.ipynb` z wypakowanego folderu (np. `Regresja_liniowa.ipynb`)


## Krok 3: Instalacja pakietów

Na samym początku notebooka (przed pierwszą komórką z kodem) dodaj nową komórkę i wklej:

```python
!pip install tensorflow==2.15 pillow==9.4.0 pandas==1.4.1 scikit-learn==1.0.2 \
    seaborn==0.11.2 plotly==5.1.0 pydot==1.4.2 graphviz==0.20.3 \
    matplotlib==3.4.3 ipywidgets==8.1.2 gdown
```

Uruchom tę komórkę (`Shift+Enter`). Instalacja potrwa 1-2 minuty.

**Uwaga:** Colab może wyświetlić ostrzeżenie o restarcie sesji — kliknij **"Restart runtime"** jeśli się pojawi, a potem uruchom komórkę instalacyjną ponownie.


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

Pliki z danymi (np. `data_houses.csv`) zostaną pobrane automatycznie przez `utils.py` (z Google Drive).

Jeśli automatyczne pobieranie nie zadziała, dane można wgrać ręcznie przez panel plików w Colabie (ikona folderu po lewej).


## Ważne informacje o Colabie

- **Sesja trwa ~12 godzin** — po tym czasie środowisko się resetuje i trzeba ponownie instalować pakiety i wgrywać pliki
- **Bezczynność ~90 minut** — Colab rozłącza sesję po ~90 minutach bez aktywności
- **Wersje pakietów** — Colab ma preinstalowane swoje wersje pakietów, które mogą się nieznacznie różnić. Komórka z `!pip install` powinna to nadpisać
- **GPU** — Colab daje darmowy dostęp do GPU. Aby go włączyć: **Runtime** → **Change runtime type** → **T4 GPU**


## Alternatywa: MyBinder

Można też otworzyć środowisko jednym kliknięciem przez MyBinder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NXTRSS/MachineLearningCourse/HEAD)

**Uwagi:** MyBinder jest wolniejszy od Colaba, budowanie środowiska może trwać 5-10 minut, a sesje mają ograniczoną pamięć i czas.
