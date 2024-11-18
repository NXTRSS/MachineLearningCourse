import pandas as pd
import traitlets
from ipywidgets import widgets
from IPython.display import display
from tkinter import Tk, filedialog
import numpy as np
from PIL import Image
import urllib
import os
import matplotlib.pyplot as plt
import io
import subprocess

DATA_FILES = {
    "catvsnotcat": {
        "file_id": "1KE3IOH0OxPI5QTeV2WI9Oi8GIVTBC8vw",
        "filename": "catvsnotcat.pkl"
    },
    "other_dataset": {
        "file_id": "1CAF1VATIvKNtQXpXo4VvVxrFi_t75Qea",
        "filename": "data_houses.csv"
    }
}

def download_file(file_id, filename):
    """
    Funkcja do pobrania pliku, jeśli jeszcze go nie ma.
    Sprawdza, czy plik istnieje, jeśli nie, pobiera go.
    """
    # Generowanie URL na podstawie ID pliku
    url = f"https://drive.google.com/uc?id={file_id}"

    # Sprawdzenie, czy trzeba zainstalować gdown
    try:
        subprocess.run(["pip", "show", "gdown"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("Instaluję gdown...")
        subprocess.run(["pip", "install", "gdown"], check=True)

    if not os.path.exists(filename):
        print(f"Plik {filename} nie istnieje, pobieram...")
        
        # Pobranie pliku przy pomocy gdown
        try:
            import gdown
            gdown.download(url, filename, quiet=False)
            print(f"Plik {filename} został pomyślnie pobrany!")
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

class SelectFilesButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(self):
        super(SelectFilesButton, self).__init__()
        # Add the selected_files trait
        self.add_traits(files=traitlets.traitlets.List())
        # Create the button.
        self.description = "Select Files"
        self.icon = "square-o"
        self.style.button_color = "orange"
        # Set on click behavior.
        self.on_click(self.select_files)

    @staticmethod
    def select_files(b):
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button 
        """
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)
        # List of selected fileswill be set to b.value
        b.files = filedialog.askopenfilename(multiple=True)

        b.description = "Files Selected"
        b.icon = "check-square-o"
        b.style.button_color = "lightgreen"

def predict_image_from_files(model_inference, my_button, classNames):
    # Get the input shape from the model
    input_shape = model_inference.input_shape
    
    # Handle flattened input vs. 3D image input
    if len(input_shape) == 2:  # Flattened input case
        IMG_SIZE = int(np.sqrt(input_shape[1] // 3))  # Calculate IMG_SIZE from flattened size
        flattening = True
    elif len(input_shape) == 4:  # 3D image input case with color channels
        IMG_SIZE = input_shape[1]
        flattening = False
    else:
        raise ValueError(f"{bcolors.BOLD}{bcolors.FAIL}Wykryto zły kształt wejścia do modelu: {model_inference.input.shape}{bcolors.ENDC}")

    if len(my_button.files) > 0:
        cols = 2 if len(my_button.files) >= 2 else 1
        rows = int(np.ceil(len(my_button.files) / 2))
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 12))
        
        if len(my_button.files) >= 2:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for image_path, axis in zip(my_button.files, axes):
            axis.axis('off')
            
            # Open image with PIL
            image = Image.open(image_path).convert('RGB')
            axis.imshow(image)
            
            # Resize the image based on model's expected input size
            resized_down = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            resized_down = np.array(resized_down) / 255.0
            
            # Flatten if the model expects flattened input
            if flattening:
                resized_down = resized_down.reshape(1, -1)
            else:
                resized_down = tf.expand_dims(resized_down, axis=0)
            
            predicted_classes = np.squeeze(model_inference.predict(resized_down))
            axis.set_title(f"Prediction: {classNames[np.argmax(predicted_classes)]}")
        
        if len(axes) > len(my_button.files):
            axes[-1].axis('off')
        
        # Show the plot with the images and predictions
        plt.show()

def predict_image_from_urls(model_inference, urls, classNames):
    # Get the input shape from the model
    input_shape = model_inference.input_shape
    
    # Handle flattened input vs. 3D image input
    if len(input_shape) == 2:  # Flattened input case
        IMG_SIZE = int(np.sqrt(input_shape[1] // 3))  # Calculate IMG_SIZE from flattened size
        flattening = True
    elif len(input_shape) == 4:  # 3D image input case with color channels
        IMG_SIZE = input_shape[1]
        flattening = False
    else:
        raise ValueError(f"{bcolors.BOLD}{bcolors.FAIL}Wykryto zły kształt wejścia do modelu: {model_inference.input.shape}{bcolors.ENDC}")
        
    cols = 2 if len(urls) >= 2 else 1
    rows = int(np.ceil(len(urls) / 2))
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 12))
    
    if len(urls) >= 2:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for photo_url, axis in zip(urls, axes):
        url_response = urllib.request.urlopen(photo_url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    
        # Load image with PIL
        image = Image.open(io.BytesIO(img_array)).convert('RGB')
        axis.imshow(image)
        
        resized_down = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        resized_down = np.array(resized_down) / 255.0
        
        if flattening:
            resized_down = resized_down.reshape(1, -1)
        else:
            resized_down = tf.expand_dims(resized_down, axis=0)
        
        predicted_classes = np.squeeze(model_inference.predict(resized_down))
        axis.set_title(f"Prediction: {classNames[np.argmax(predicted_classes)]}")
        axis.axis('off')
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
