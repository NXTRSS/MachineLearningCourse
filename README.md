# Zajęcia z Uczenia Maszynowego

Poniżej przedstawiam instrukcję niezbędnych kroków do wykonania przed zajęciami


## Wymagania sprzętowe

Wirtualna Maszyna z Ubuntu 20.04, **RAM** minimum 5918MB najlepiej 8192MB, pamięć 30GB dla wygody 40GB

## Zainstalowanie Minicondy
Pomoże nam ona w zarządzaniu środowiskami pythonowymi oraz w ściaganiu niezbędnych paczek (np. Tensor Flow)
```bash
sudo apt-get update

sudo apt-get install curl
```
Wymagane jest ściagnięcie odpowiedniej wersji Minicondy -  odpowiadającej systemowi operacyjnemu (Linux w naszym przypadku) oraz architekturze procesora.\
\
**Proszę sprawdzić, który link będzie dla państwa odpowiedni pod tym adresem**: https://docs.conda.io/en/latest/miniconda.html#linux-installers\
Załóżmy, że będzie to procesor Intela 64-bit. W takiej sytuacji korzystamy z linku: *https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh*
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
```
W przypadku procesorów Mac od Appla M1 byłoby to:
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-aarch64.sh
```
Następnie, po ściągnięciu plików do instalacji Minicondy, wystarczy uruchomić proces instalacji\ *(P.S. Najprawdopodobniej jeśli dziwne krzaczki ściagały się podczas wywołania powyższej lini, zły link został podany)*
```bash
# Oczywiście poniższy plik musi się zgadzać z tym, który został ściagnięty
# czyli dla procesora M1 końcówka byłaby Linux-aarch64.sh
bash Miniconda3-py38_4.11.0-Linux-x86_64.sh
```
Następnie trzeba przejść przez instalację podając Enter i/lub *yes* w terminalu.\
**Ważne: Po powyższych krokach proszę zrestartować terminal**

### Problemy z zainstalowaniem Minicondy
Jeśli będą jakieś problemy z zainstalowaniem Minicondy zgodnie z powyższą instrukcją bardzo proszę odnieść się do dokumentacji: [Miniconda documentation](https://docs.conda.io/en/latest/miniconda.html#miniconda)\
Jeśli nadal występują problemy, w krytycznej sytuacji, proszę sprobować ściągnąć Anacondę zgodnie z dokumentacją: [Anaconda Linux](https://docs.anaconda.com/anaconda/install/linux/), w takiej sytuacji trzeba nadal zwrócić uwagę na architekturę procesora.
## Ściągnięcie plików z kodami
Jeśli zainstalowanie Minicondy przebiegło pomyślę oraz **zrestartowali państwo terminal** powinno być widoczne słówko *base* w nawiasach w naszym terminalu (przykład):
```bash
(base) kamil@ubuntu:~$
```
Ta nazwa w nawiasie to właśnie nazwa naszego środowiska python - w tej sytuacji domyślnie jest to *base*.\
Stwórzmy nowe foldery aby łatwiej poruszać się po plikach:
```bash
cd ~
mkdir -p MachineLearningCodes
cd MachineLearningCodes
```
Teraz do folderu ściągnijmy pliki z GitHub potrzebne podczas zajęć
```bash
git clone https://github.com/NXTRSS/MachineLearningCourse
```
*P.S. Można też wejść w dostarczony link i ściągnąć dane przez wygenerowanie archiwum .zip a następnie wypakowanie go w naszym folderze*
## Stworzenie środowiska python
Jeśli ściągnięcie plików przebiegło poprawnie w naszym folderze powinien znajdować się plik *environment.yaml* i za jego pomocą stworzymy nowe środowisko pythonowe o nazwie **ml**:
```bash
cd MachineLearningCourse

conda update -n base -c defaults conda -y

conda env create -f environment.yaml

conda activate ml
```
**Jeśli powyższe komendy nie są w stanie być wykonane polecam uruchomić poniższe:**\
*Można też posiłkować się paczkami i wersjami zawartymi w plikach environment.yaml oraz requirements.txt (w szczególności gdy ktoś korzysta z venv zamiast z condy)*
```bash
conda update -n base -c defaults conda -y

conda create -n ml python=3.9.7 -y
conda activate ml

pip install tensorflow==2.15 graphviz==0.20.3
conda install pillow=9.4.0 pandas=1.4.1 scikit-learn=1.0.2 -y 
conda install seaborn=0.11.2 plotly=5.1.0 pydot=1.4.2 jupyterlab=4.2.5 matplotlib=3.4.3 ipywidgets=8.1.2 -y
```
Po aktywacji nowego środowiska zamiast *base* powinno być widoczne *ml* w naszym terminalu (przykład):
```bash
(ml) kamil@ubuntu:~$
```
**Proszę pamiętać aby zawsze aktywować to środowisko po wznowieniu pracy na komputerze!**\
Proszę wywołać poniższe linijki aby aktywować kilka dodatkowych ustawień:
```bash
python -m ipykernel install --user --name ml --display-name "Python (ml)"
```
## Weryfikacja środowiska Python

Aby upewnić się, że środowisko Pythonowe zostało skonfigurowane poprawnie, przygotowany został specjalny skrypt weryfikacyjny. Proszę wykonać poniższe kroki:

1. Upewnij się, że aktywowane jest środowisko `ml`:
   ```bash
   conda activate ml
   ```
2. W folderze MachineLearningCourse uruchom skrypt weryfikacyjny:
   ```bash
   python verify_env.py
   ```
3. Skrypt sprawdzi wersję Pythona oraz dostępność wymaganych bibliotek. Jeśli wszystko zostało zainstalowane poprawnie, skrypt wyświetli komunikat: `Environment verification: OK.`
   
   Przykładowy zwrot z terminala po uruchomieniu skryptu:
   
   ```bash
   (ml) kamil@ubuntu:~/MachineLearningCodes/MachineLearningCourse$ python verify_env.py

   Checking Python version...

   Python Version: OK (Version 3.9.7)

   Checking installed packages...

   Package tensorflow: OK (Version 2.15.0)
   Package pillow: OK (Version 9.4.0)
   Package pandas: OK (Version 1.4.1)
   Package scikit-learn: OK (Version 1.0.2)
   Package seaborn: OK (Version 0.11.2)
   Package plotly: OK (Version 5.1.0)
   Package pydot: OK (Version 1.4.2)
   Package jupyterlab: OK (Version 4.2.5)
   Package matplotlib: OK (Version 3.4.3)
   Package ipywidgets: OK (Version 8.1.2)

   Checking system-level packages...

   Package graphviz: OK (Installed via conda)

   Environment verification: OK
   ```
Jeśli pojawią się błędy lub ostrzeżenia, skrypt wskaże brakujące paczki lub niezgodności w wersjach. Proszę upewnić się, że wszystkie wymagane pakiety zostały zainstalowane zgodnie z instrukcjami. Oczywiście jest szansa, że paczka w innej wersji będzie działała, lecz nie ma pewności. Przy złej wersji zwrot z terminala mógłby wyglądać tak:
```bash
Checking Python version...

WARNING: Python Version: 3.9.10 (Expected: 3.9.7)

Checking installed packages...

Package tensorflow: OK (Version 2.15.0)
Package pillow: WARNING (Installed Version: 9.3.0, Expected: 9.4.0)
Package pandas: OK (Version 1.4.1)
Package scikit-learn: OK (Version 1.0.2)
Package seaborn: WARNING (Installed Version: 0.11.1, Expected: 0.11.2)
Package plotly: OK (Version 5.1.0)
Package pydot: OK (Version 1.4.2)
Package jupyterlab: OK (Version 4.2.5)
Package matplotlib: OK (Version 3.4.3)
Package ipywidgets: OK (Version 8.1.2)

Checking system-level packages...

Package graphviz: OK (Installed via conda)

Environment verification: WARNINGS DETECTED.

Some packages have mismatched versions. While the environment is functional, it is recommended to align package versions to ensure consistency.
```

### Co zrobić gdy nadal nie można prawidłowo stworzyć środowiska?
W takiej sytuacji na zajęciach będzie można:
1. Otworzyć instację w MyBinder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NXTRSS/MachineLearningCourse/HEAD)
2. Otworzyć Jupyter Notebook w Colab - wystarczy otworzyć nasz Google Drive i otworzyć plik .ipynb za pomocą Colab

## Rozpoczęcie (oraz wznowienie pracy)
Przy każdym wznowieniu pracy (ponownym odpaleniu komputera i maszyny wirtualnej) proszę wejście do odpowiedniego folderu:
```bash
cd MachineLearningCodes/MachineLearningCourse
```
 zaktywować środowisko o nazwie *ml*:
```bash
conda activate ml
```
A następnie wywołać narzędzie **Jupyter Notebook** w ramach **Jupyter Lab**, które jest webową aplikacją, na której będziemy pracować na naszych zajęciach:
```bash
jupyter lab
```
Po aktywacji powyższej komendy otworzy się przeglądarka a w niej nasze pliki.

**Ważne: w narzędziu Jupyter Notebook do uruchamiania przygotowanych skryptów trzeba będzie "wyklikać" kernel *Python (ml)* - będzie to pokazane na zajęciach.**

Do działania plików niezbędne będą również zbiory danych - umieszczone na dysku Google
