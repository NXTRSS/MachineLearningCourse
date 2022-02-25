import os
import io
import pyAesCrypt
import pandas as pd
import pickle
import traitlets
from ipywidgets import widgets
from IPython.display import display
from tkinter import Tk, filedialog


def decrypt_file(file_path, password):
    if password is None or password == '':
        raise ValueError(f'{bcolors.BOLD}{bcolors.FAIL}Please provide password')
        
    bufferSize = 64 * 1024
    encFileSize = os.stat(file_path).st_size

    fIn = open(file_path, 'rb')
    fDec = io.BytesIO()

    pyAesCrypt.decryptStream(fIn, fDec, password, bufferSize, encFileSize)
    
    return fDec


def decrypt_pandas(file_path, password):        
    fDec = decrypt_file(file_path, password)

    s=str(fDec.getvalue(),'utf-8')

    data = io.StringIO(s) 
    return pd.read_table(data, sep=',')


def decrypt_pickle(file_path, password):
    fDec = decrypt_file(file_path, password)

    return pickle.loads(fDec.getvalue())


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