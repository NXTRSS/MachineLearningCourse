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