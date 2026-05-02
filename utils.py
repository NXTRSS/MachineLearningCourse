import pandas as pd
from ipywidgets import widgets
from IPython.display import display
import numpy as np
from PIL import Image
import urllib
import os
import matplotlib.pyplot as plt
import io
import tensorflow as tf

DATA_FILES = {
    "catvsnotcat": {
        "file_id": "1KE3IOH0OxPI5QTeV2WI9Oi8GIVTBC8vw",
        "filename": "catvsnotcat.pkl"
    },
    "data_houses": {
        "file_id": "1CAF1VATIvKNtQXpXo4VvVxrFi_t75Qea",
        "filename": "data_houses.csv"
    }
}

def download_file(file_id, filename):
    """
    Funkcja do pobrania pliku, jeśli jeszcze go nie ma.
    Sprawdza, czy plik istnieje, jeśli nie, pobiera go.
    """
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(filename):
        print(f"Plik {filename} nie istnieje, pobieram...")
        try:
            import gdown
            gdown.download(url, filename, quiet=False)
            print(f"Plik {filename} został pomyślnie pobrany!")
        except ImportError:
            print(f"ERROR: Pakiet 'gdown' nie jest zainstalowany. Zainstaluj go: pip install gdown")
        except Exception as e:
            print(f"Błąd podczas pobierania pliku {filename}: {e}")
    else:
        print(f"Plik {filename} już istnieje, nie trzeba pobierać.")

def check_and_download_data(files_to_check=None):
    """
    Sprawdza, czy określone dane są dostępne i pobiera je, jeśli to konieczne.
    
    Parametr:
        files_to_check (list): Lista nazw plików do sprawdzenia i pobrania. Jeśli None, sprawdzane są wszystkie pliki w DATA_FILES.
    """
    if files_to_check is None:
        # Jeśli nie podano listy, sprawdzają się wszystkie pliki
        files_to_check = DATA_FILES.keys()
    
    for key in files_to_check:
        if key in DATA_FILES:
            data = DATA_FILES[key]
            download_file(data["file_id"], data["filename"])
        else:
            print(f"Błąd: {key} nie jest zdefiniowany w DATA_FILES.")

class SelectFilesButton(widgets.FileUpload):
    """Przycisk wyboru zdjęć działający bezpośrednio w przeglądarce.

    Zastępuje poprzednią implementację opartą na tkinter, która nie działała
    w środowiskach bez GUI (Docker, JupyterHub, VS Code, itp.).

    Użycie:
        btn = SelectFilesButton()
        display(btn)
        # po wyborze pliku(ów):
        for file_info in btn.value:
            img = Image.open(io.BytesIO(file_info['content'].tobytes()))
    """

    def __init__(self):
        super().__init__(
            accept='image/*',
            multiple=True,
            description='Wybierz zdjęcia',
            button_style='warning',
            layout=widgets.Layout(width='auto'),
        )

def _model_input_geometry(model_inference):
    input_shape = model_inference.input_shape
    if len(input_shape) == 2:
        return int(np.sqrt(input_shape[1] // 3)), True
    if len(input_shape) == 4:
        return input_shape[1], False
    raise ValueError(
        f"{bcolors.BOLD}{bcolors.FAIL}Wykryto zły kształt wejścia do modelu: "
        f"{model_inference.input_shape}{bcolors.ENDC}"
    )


def _iter_uploaded_images(my_button):
    """Iteruje po obrazkach z FileUpload — kompatybilne z ipywidgets 7.x i 8.x."""
    value = my_button.value

    # ipywidgets 8.x: tuple/list of dicts {name, type, size, content: memoryview}
    if isinstance(value, (tuple, list)) and len(value) > 0:
        for file_info in value:
            content = file_info['content']
            if hasattr(content, 'tobytes'):
                content = content.tobytes()
            yield file_info['name'], content
        return

    # ipywidgets 7.x: {filename: {metadata, content: bytes}}
    if isinstance(value, dict) and len(value) > 0:
        for fname, meta in value.items():
            yield fname, meta['content']
        return

    # Fallback: sprawdź atrybut .data (starsze wersje ipywidgets)
    data = getattr(my_button, 'data', None)
    if data:
        if isinstance(data, (list, tuple)):
            for i, raw in enumerate(data):
                yield f"image_{i}", raw
        elif isinstance(data, dict):
            for fname, raw in data.items():
                yield fname, raw


def _predict_one(model_inference, image, classNames, IMG_SIZE, flattening):
    resized_down = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    resized_down = np.array(resized_down) / 255.0
    if flattening:
        resized_down = resized_down.reshape(1, -1)
    else:
        resized_down = np.expand_dims(resized_down, axis=0)
    predicted_classes = np.squeeze(model_inference.predict(resized_down, verbose=0))
    return classNames[int(np.argmax(predicted_classes))], float(np.max(predicted_classes))


def predict_image_from_files(model_inference, my_button, classNames):
    IMG_SIZE, flattening = _model_input_geometry(model_inference)
    files = list(_iter_uploaded_images(my_button))

    if not files:
        print(f"{bcolors.WARNING}Najpierw wybierz zdjęcie/zdjęcia za pomocą przycisku powyżej.{bcolors.ENDC}")
        return

    cols = 2 if len(files) >= 2 else 1
    rows = int(np.ceil(len(files) / 2))
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 6 * rows))
    axes = axes.flatten() if len(files) >= 2 else [axes]

    for (fname, raw_bytes), axis in zip(files, axes):
        image = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
        axis.imshow(image)
        axis.axis('off')
        label, prob = _predict_one(model_inference, image, classNames, IMG_SIZE, flattening)
        axis.set_title(f"Prediction: {label} ({prob:.0%})")

    for extra_axis in axes[len(files):]:
        extra_axis.axis('off')
    plt.show()


def predict_image_from_urls(model_inference, urls, classNames):
    IMG_SIZE, flattening = _model_input_geometry(model_inference)

    cols = 2 if len(urls) >= 2 else 1
    rows = int(np.ceil(len(urls) / 2))
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 6 * rows))
    axes = axes.flatten() if len(urls) >= 2 else [axes]

    for photo_url, axis in zip(urls, axes):
        # Wikimedia i część CDN zwracają 403 bez User-Agent
        req = urllib.request.Request(photo_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            img_bytes = response.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        axis.imshow(image)
        axis.axis('off')
        label, prob = _predict_one(model_inference, image, classNames, IMG_SIZE, flattening)
        axis.set_title(f"Prediction: {label} ({prob:.0%})")

    for extra_axis in axes[len(urls):]:
        extra_axis.axis('off')
    plt.show()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
