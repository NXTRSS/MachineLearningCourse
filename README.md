# Zajęcia z Uczenia Maszynowego WSB

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
mkdir WSB/ml -p
cd WSB/ml
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

conda env create -f environment.yaml

conda activate ml
```
Po aktywacji nowego środowiska zamiast *base* powinno być widoczne *ml* w naszym terminalu (przykład):
```bash
(ml) kamil@ubuntu:~$
```
**Proszę pamiętać aby zawsze aktywować to środowisko po wznowieniu pracy na komputerze!**\
Proszę wywołać poniższe linijki aby aktywować kilka dodatkowych ustawień:
```bash
jupyter nbextension enable codefolding/edit
python -m ipykernel install --user --name ml --display-name "Python (ml)”
```
## Rozpoczęcie (oraz wznowienie pracy)
Przy każdym wznowieniu pracy (ponownym odpaleniu komputera i maszyny wirtualnej) proszę wejście do odpowiedniego folderu:
```bash
cd WSB/ml/MachineLearningCourse
```
 zaktywować środowisko o nazwie *ml*:
```bash
conda activate ml
```
A następnie wywołać narzędzie **Jupyter Notebook**, które jest webową aplikacją, na której będziemy pracować na naszych zajęciach:
```bash
jupyter notebook
```
Po aktywacji powyższej komendy otworzy się przeglądarka a w niej nasze pliki.\
**Ważne: w narzędziu Jupyter Notebook do uruchamiania przygotowanych skryptów trzeba będzie "wyklikać" kernel *Python (ml)* - będzie to pokazane na zajęciach.**\
Do działania plików niezbędne będą również zbiory danych - umieszczone na dysku Google
