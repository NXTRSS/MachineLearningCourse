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


def _is_logreg(model):
    """Sprawdza czy model to dict regresji logistycznej (z kluczami 'w' i 'b')."""
    return isinstance(model, dict) and "w" in model and "b" in model


def _normalize_models(models):
    """Zamienia input na dict {nazwa: model}.

    Obsługuje:
      - dict regresji logist. (z 'w','b') → {"Logistic Regression": dict}
      - pojedynczy model Keras             → {model.name: model}
      - listę/krotkę                       → {model.name: model, ...}
      - dict {nazwa: model}                → bez zmian
    """
    if _is_logreg(models):
        return {"Logistic Regression": models}
    if isinstance(models, dict):
        return models
    if isinstance(models, (list, tuple)):
        result = {}
        for m in models:
            name = "Logistic Regression" if _is_logreg(m) else m.name
            result[name] = m
        return result
    return {models.name: models}


def _predict_one(model, image, classNames):
    """Predykcja jednego obrazka jednym modelem (sam wykrywa geometrię)."""
    if _is_logreg(model):
        # Regresja logistyczna: dict z 'w' i 'b'
        w, b = model["w"], model["b"]
        IMG_SIZE = int(np.sqrt(w.shape[0] // 3))
        resized = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        resized = np.array(resized) / 255.0
        resized = resized.reshape(-1, 1)
        A = 1 / (1 + np.exp(-(np.dot(w.T, resized) + b)))
        prob = float(np.squeeze(A))
        predicted_class = 1 if prob > 0.5 else 0
        confidence = prob if predicted_class == 1 else 1 - prob
        label = classNames[predicted_class] if isinstance(classNames, (list, tuple)) else classNames.get(predicted_class, str(predicted_class))
        return label, confidence

    # Model Keras
    IMG_SIZE, flattening = _model_input_geometry(model)
    resized = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    resized = np.array(resized) / 255.0
    if flattening:
        resized = resized.reshape(1, -1)
    else:
        resized = np.expand_dims(resized, axis=0)
    predicted_classes = np.squeeze(model.predict(resized, verbose=0))
    idx = int(np.argmax(predicted_classes))
    label = classNames[idx] if isinstance(classNames, (list, tuple)) else classNames.get(idx, str(idx))
    return label, float(np.max(predicted_classes))


def _build_title(models_dict, image, classNames):
    """Buduje tytuł z predykcjami wszystkich modeli."""
    if len(models_dict) == 1:
        model = list(models_dict.values())[0]
        label, prob = _predict_one(model, image, classNames)
        return f"Prediction: {label} ({prob:.0%})"
    parts = []
    for name, model in models_dict.items():
        label, prob = _predict_one(model, image, classNames)
        parts.append(f"{name}: {label} ({prob:.0%})")
    return "  |  ".join(parts)


def predict_image_from_files(models, my_button, classNames):
    """Predykcja z przesłanych plików.

    models: pojedynczy model, lista modeli, lub dict {nazwa: model}
    """
    models_dict = _normalize_models(models)
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
        axis.set_title(_build_title(models_dict, image, classNames), fontsize=12)

    for extra_axis in axes[len(files):]:
        extra_axis.axis('off')
    plt.show()


def predict_image_from_urls(models, urls, classNames):
    """Predykcja z URLi.

    models: pojedynczy model, lista modeli, lub dict {nazwa: model}
    """
    models_dict = _normalize_models(models)

    cols = 2 if len(urls) >= 2 else 1
    rows = int(np.ceil(len(urls) / 2))
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 6 * rows))
    axes = axes.flatten() if len(urls) >= 2 else [axes]

    for photo_url, axis in zip(urls, axes):
        req = urllib.request.Request(photo_url, headers={'User-Agent': 'Mozilla/5.0'})
        for _attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    img_bytes = response.read()
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 502, 503, 504) and _attempt < 2:
                    import time; time.sleep(2 * (_attempt + 1))
                else:
                    raise
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        axis.imshow(image)
        axis.axis('off')
        axis.set_title(_build_title(models_dict, image, classNames), fontsize=12)

    for extra_axis in axes[len(urls):]:
        extra_axis.axis('off')
    plt.show()

from sklearn.decomposition import PCA
from collections import namedtuple
import matplotlib.lines as mlines
from matplotlib import cm


def to_2d(embeddings, pca_model=None):
    if pca_model is None:
        pca_model = PCA(n_components=2, whiten=True)
        pca_model.fit(embeddings)
    return pca_model.transform(embeddings)


def annotated_scatter(points, names, color='blue'):
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    plt.scatter(x_coords, y_coords, c=color)
    for label, x, y in zip(names, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() - .5, x_coords.max() + .5)
    plt.ylim(y_coords.min() - .5, y_coords.max() + .5)


def plot_embeddings(embeddings, names, color='blue', show=True, pca_model=None):
    X_train = np.array([embeddings[k] for k in names])
    embeddings_2d = to_2d(X_train, pca_model=pca_model)

    annotated_scatter(embeddings_2d, names, color)
    plt.grid()

    if show:
        plt.show()


LinearSubs = namedtuple('LinearSubs', ('word_pair', 'name'))


def plot_linear_substructures(linear_subs, embeddings, pca_model=None):
    embeddings_matrix = [embeddings[p] for ls in linear_subs for p in ls.word_pair]
    embeddings_matrix = np.array(embeddings_matrix)
    pair_names = [p for ls in linear_subs for p in ls.word_pair]
    ls_names = [ls.name for ls in linear_subs]
    embeddings_2d = to_2d(embeddings_matrix, pca_model=pca_model)
    annotated_scatter(embeddings_2d,
                      pair_names,
                      cm.Set1.colors[:len(embeddings_2d)])

    for i in range(0, len(embeddings_2d), 2):
        p1 = embeddings_2d[i]
        p2 = embeddings_2d[i + 1]
        center = [(p1[j] + p2[j]) / 2 + .04 for j in range(2)]

        plt.plot(*zip(p1, p2), '--')
        plt.annotate(ls_names[i // 2],
                     xy=center,
                     xytext=(0, 0), textcoords='offset points')


def glove_most_similar(embeddings, positive=None, negative=None, topn=10):
    vec = np.zeros_like(list(embeddings.values())[0])
    query_words = set()
    for w in (positive or []):
        vec = vec + embeddings[w]
        query_words.add(w)
    for w in (negative or []):
        vec = vec - embeddings[w]
        query_words.add(w)

    norms = np.array([np.linalg.norm(v) for v in embeddings.values()])
    dots = np.array([np.dot(vec, v) for v in embeddings.values()])
    sims = dots / (norms * np.linalg.norm(vec) + 1e-10)

    words = list(embeddings.keys())
    ranked = sorted(zip(words, sims), key=lambda x: -x[1])
    return [(w, float(s)) for w, s in ranked if w not in query_words][:topn]


_TOKEN_COLORS = [
    '#BBDEFB', '#C8E6C9', '#FFE0B2', '#F8BBD0',
    '#D1C4E9', '#B2EBF2', '#FFF9C4', '#FFCCBC',
    '#DCEDC8', '#F0F4C3', '#B3E5FC', '#E1BEE7',
]

def show_tokens(text, label='', model='gpt-4o'):
    import html as _html
    import tiktoken
    from IPython.display import display, HTML

    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    decoded = [enc.decode([t]) for t in tokens]

    spans = []
    for i, tok in enumerate(decoded):
        color = _TOKEN_COLORS[i % len(_TOKEN_COLORS)]
        visible = tok.replace(' ', '␣').replace('\n', '↵')
        visible = _html.escape(visible)
        spans.append(
            f'<span style="background:{color}; padding:2px 4px; '
            f'border-radius:3px; margin:1px; display:inline-block; '
            f'font-family:monospace; font-size:14px; '
            f'border:1px solid rgba(0,0,0,0.1);">{visible}</span>'
        )

    header = f'<b style="font-size:13px; color:#555;">{label}</b><br>' if label else ''
    html = (
        f'<div style="margin:6px 0;">'
        f'{header}'
        f'<code style="font-size:13px;">{_html.escape(text)}</code><br>'
        f'<span style="font-size:12px; color:#888;">{len(tokens)} tokenów:</span> '
        f'{" ".join(spans)}'
        f'</div>'
    )
    display(HTML(html))


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


def display_generated_city(prompt):
    """Wyświetla wygenerowaną nazwę miasta z Modelu Języka, ładnie sformatowaną z emoji 🏘️.

    Argument `prompt` to pełny ciąg z tokenami specjalnymi % i ! — np. '%Babciowice!'.
    """
    from IPython.display import display, Markdown
    display(Markdown(f"## 🏘️ **Wygenerowana nazwa miasta:** `{prompt[1:-1].title()}`"))


def visualize_probabilities(probabilities, step, tokenizer, prompt, chosen_idx, show=True):
    """Rysuje rozkład prawdopodobieństw kolejnego znaku w Modelu Języka.

    Słupek wybranego znaku (`chosen_idx`) jest oznaczony na czerwono. Jeżeli
    sampling stochastyczny wybrał coś innego niż argmax — argmax jest dodatkowo
    oznaczony na pomarańczowo, a druga linia tytułu informuje o tym fakcie.

    Argumenty:
        probabilities  - 1D ndarray z prawdopodobieństwami dla wszystkich tokenów (z paddingiem 0)
        step           - numer kroku w generacji (do tytułu)
        tokenizer      - keras Tokenizer z mapą `index_word`
        prompt         - dotychczasowy wynik generacji (do tytułu)
        chosen_idx     - indeks tokena wybranego w tym kroku
        show           - czy od razu pokazać wykres (False przy zbieraniu wielu kroków)
    """
    # pomijamy pozycję 0 (padding) — wszystkie indeksy znaków zaczynają się od 1
    characters = [char for idx, char in tokenizer.index_word.items() if idx > 0]
    probs = probabilities[1:]

    chosen_pos = int(chosen_idx) - 1
    argmax_pos = int(probs.argmax())
    chosen_char = tokenizer.index_word.get(int(chosen_idx), '?')
    chosen_prob = float(probs[chosen_pos]) if 0 <= chosen_pos < len(probs) else 0.0

    # kolory słupków: chosen — czerwony, argmax (jeśli inny) — pomarańczowy, reszta — szary
    colors = ['#cfd8dc'] * len(characters)
    if argmax_pos != chosen_pos and 0 <= argmax_pos < len(colors):
        colors[argmax_pos] = '#f39c12'
    if 0 <= chosen_pos < len(colors):
        colors[chosen_pos] = '#e74c3c'

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(characters, probs, color=colors, edgecolor='white', linewidth=0.5)

    # tytuł — pierwsza linia neutralna, druga linia kolorowa zależnie od decyzji
    line1 = f'Krok {step} — dotychczasowy wynik: {prompt}'
    if argmax_pos == chosen_pos:
        line2 = f'Wybrano "{chosen_char}" (P={chosen_prob:.3f}) — najwyższe prawdopodobieństwo'
        line2_color = '#1e8449'
    else:
        argmax_char = tokenizer.index_word.get(argmax_pos + 1, '?')
        argmax_prob = float(probs[argmax_pos])
        line2 = (f'Wybrano "{chosen_char}" (P={chosen_prob:.3f}) — '
                 f'NIE najwyższe; argmax był "{argmax_char}" (P={argmax_prob:.3f})')
        line2_color = '#c0392b'

    ax.set_title(line1, fontsize=12, fontweight='bold', loc='left', pad=22)
    ax.text(0, 1.02, line2, transform=ax.transAxes, fontsize=10, color=line2_color)

    ax.set_xlabel('Znaki')
    ax.set_ylabel('Prawdopodobieństwo')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()


# ─────────────────────────── LLM auto-detect ────────────────────────────

PREFERRED_MODELS = [
    "qwen3.5:27b", "qwen3.5:9b", "qwen3.5:4b",
    "qwen3:14b", "qwen3:8b",
    "gemma4:12b", "gemma4:4b",
    "gemma3:27b", "gemma3:12b", "gemma3:8b",
    "qwen2.5:14b", "qwen2.5:7b",
    "gemma2:9b", "mistral:7b", "llama3.1:8b",
    "qwen3.5:2b", "qwen3.5:0.8b",
    "qwen3:4b", "qwen3:1.7b", "qwen3:0.6b",
    "gemma4:e4b", "gemma4:e2b",
    "gemma3:4b", "gemma3:1b", "qwen2.5:3b",
]


def detect_ollama(base_url="http://localhost:11434"):
    import requests
    headers = {"ngrok-skip-browser-warning": "true"} if "ngrok" in base_url else {}
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=3, headers=headers)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def detect_lmstudio(base_url="http://localhost:1234"):
    import requests
    try:
        r = requests.get(f"{base_url}/api/v1/models", timeout=3)
        if r.status_code == 200:
            models = r.json().get("models", [])
            return [m["key"] for m in models if m.get("type") == "llm"]
    except Exception:
        pass
    return []


def _try_launch_lms():
    import shutil, subprocess, time
    if not shutil.which("lms"):
        return False
    try:
        subprocess.Popen(
            ["lms", "server", "start"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("  Uruchamiam LM Studio (`lms server start`)...")
        for _ in range(6):
            time.sleep(2)
            if detect_lmstudio():
                return True
        return False
    except Exception:
        return False


def pick_best_model(available_models, preferred=None):
    preferred = preferred or PREFERRED_MODELS
    for p in preferred:
        pname = p.split(":")[0]
        for a in available_models:
            if pname in a:
                return a
    return available_models[0] if available_models else None


def connect_llm(instructor_server="http://192.168.1.100:11434"):
    """Wykryj działający LLM i zwróć (OpenAI_client, model_name, base_url).

    Kolejność prób: LM Studio → auto-launch lms → Ollama → serwer prowadzącego.
    Zwraca (None, None, None) jeśli nic nie znalezione.
    """
    from openai import OpenAI

    # 1) LM Studio
    print("Szukam LM Studio (port 1234)...")
    models = detect_lmstudio()
    if not models:
        models = _try_launch_lms() and detect_lmstudio()
    if models:
        model = pick_best_model(models) or models[0]
        url = "http://localhost:1234"
        client = OpenAI(base_url=f"{url}/v1", api_key="lm-studio")
        print(f"✓ LM Studio! Model: {model}")
        return client, model, url

    # 2) Ollama lokalna
    print("  LM Studio niedostępne.\nSzukam lokalnej Ollamy (port 11434)...")
    models = detect_ollama()
    if models:
        model = pick_best_model(models)
        url = "http://localhost:11434"
        client = OpenAI(base_url=f"{url}/v1", api_key="ollama")
        print(f"✓ Lokalna Ollama! Model: {model}")
        return client, model, url

    # 3) Serwer prowadzącego
    if instructor_server:
        print("  Ollama niedostępna.\nPróbuję serwer prowadzącego...")
        models = detect_ollama(instructor_server)
        if models:
            model = pick_best_model(models)
            client = OpenAI(base_url=f"{instructor_server}/v1", api_key="ollama")
            print(f"✓ Serwer prowadzącego! Model: {model}")
            return client, model, instructor_server

    print("✗ Brak dostępnego LLM-a! Zainstaluj LM Studio lub Ollamę (setup_local_llm.ipynb).")
    return None, None, None


# ─────────────────────────── ensure_package ───────────────────────────

def ensure_package(pip_name, import_name=None):
    """Sprawdź czy pakiet jest dostępny; jeśli nie — zainstaluj.

    Działa z uv (Plan A), Docker/pip (Plan B) i Google Colab (Plan C).
    """
    import subprocess, sys, shutil
    import_name = import_name or pip_name
    try:
        __import__(import_name)
        return
    except ImportError:
        pass

    print(f"Instaluję {pip_name}...")

    # 1) uv pip install — środowisko uv (Plan A)
    if shutil.which("uv"):
        try:
            subprocess.check_call(
                ["uv", "pip", "install", "--quiet", pip_name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # 2) python -m pip — Docker, Colab, zwykły venv
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise RuntimeError(
        f"Nie udało się zainstalować {pip_name}. "
        f"Spróbuj ręcznie: uv pip install {pip_name}  lub  pip install {pip_name}"
    )
