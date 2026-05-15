import pandas as pd
from ipywidgets import widgets
from IPython.display import display
import numpy as np
from PIL import Image
import urllib
import os
os.environ.setdefault("USE_TF", "0")
import matplotlib.pyplot as plt
import io

_GITHUB_RAW = "https://raw.githubusercontent.com/NXTRSS/MachineLearningCourse/main"
_GITHUB_RELEASE = "https://github.com/NXTRSS/MachineLearningCourse/releases/download/data-v1"

DATA_FILES = {
    "catvsnotcat": {
        "url": f"{_GITHUB_RELEASE}/catvsnotcat.pkl",
        "filename": "catvsnotcat.pkl",
    },
    "data_houses": {
        "url": f"{_GITHUB_RAW}/data_houses.csv",
        "filename": "data_houses.csv",
    },
    "prng_miejscowosci": {
        "url": f"{_GITHUB_RAW}/PRNG_MIEJSCOWOSCI_05_2021.csv",
        "filename": "PRNG_MIEJSCOWOSCI_05_2021.csv",
    },
}

def download_file(url, filename):
    """
    Pobiera plik z podanego URL, jeśli jeszcze go nie ma na dysku.
    Używa requests (streaming) — działa na local env, Docker i Colab.
    """
    if not os.path.exists(filename):
        print(f"Plik {filename} nie istnieje, pobieram...")
        try:
            import requests as _req
            resp = _req.get(url, stream=True, allow_redirects=True, timeout=300)
            resp.raise_for_status()
            # content-length bywa niedostępny lub zaniżony (gzip) → progress opcjonalny
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(filename, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):   # 1 MB
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and total >= downloaded:
                        pct = downloaded * 100 // total
                        print(f"\r  {pct}% ({downloaded >> 20}/{total >> 20} MB)", end="", flush=True)
                    else:
                        print(f"\r  {downloaded >> 20} MB", end="", flush=True)
            print(f"\nPlik {filename} został pomyślnie pobrany!")
        except Exception as e:
            # Usuń częściowo pobrany plik
            if os.path.exists(filename):
                os.remove(filename)
            print(f"Błąd podczas pobierania pliku {filename}: {e}")
    else:
        print(f"Plik {filename} już istnieje, nie trzeba pobierać.")

def check_and_download_data(files_to_check=None):
    """
    Sprawdza, czy określone dane są dostępne i pobiera je, jeśli to konieczne.

    Parametr:
        files_to_check (list): Lista nazw plików do sprawdzenia i pobrania.
                               Jeśli None, sprawdzane są wszystkie pliki w DATA_FILES.
    """
    if files_to_check is None:
        files_to_check = DATA_FILES.keys()

    for key in files_to_check:
        if key in DATA_FILES:
            data = DATA_FILES[key]
            download_file(data["url"], data["filename"])
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
    """Buduje tytuł z predykcjami wszystkich modeli (każdy w nowej linii)."""
    if len(models_dict) == 1:
        model = list(models_dict.values())[0]
        label, prob = _predict_one(model, image, classNames)
        return f"Prediction: {label} ({prob:.0%})"
    parts = []
    for name, model in models_dict.items():
        label, prob = _predict_one(model, image, classNames)
        parts.append(f"{name}: {label} ({prob:.0%})")
    return "\n".join(parts)


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


def plot_training_comparison(histories, figsize=None):
    """Porównanie accuracy i loss wielu modeli obok siebie, ze wspólną osią Y.

    histories: dict {nazwa: history_obiekt_lub_None}
        Modele z wartością None są pomijane (jeszcze nie wytrenowane).
        history może być obiektem Keras (history.history) lub dict.

    Przykład użycia w notebooku:
        plot_training_comparison({
            'FFNN': globals().get('history_model'),
            'Dropout': globals().get('history_model_reg'),
            'CNN': globals().get('history_model_cnn'),
        })
    """
    # Filtruj None i wyciągnij dict z history
    items = []
    for name, h in histories.items():
        if h is None:
            continue
        hd = h.history if hasattr(h, 'history') else h
        items.append((name, hd))

    if not items:
        print("Brak dostępnych wyników do wyświetlenia.")
        return

    n = len(items)
    if figsize is None:
        figsize = (6 * n, 8)

    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=figsize, squeeze=False)

    # Wspólne zakresy osi Y
    all_acc = [v for _, hd in items for v in hd['accuracy'] + hd['val_accuracy']]
    all_loss = [v for _, hd in items for v in hd['loss'] + hd['val_loss']]
    acc_min = 0
    acc_max = 1
    loss_max = max(all_loss) * 1.05

    colors = {'train': '#1f77b4', 'val': '#ff7f0e'}

    for col, (name, hd) in enumerate(items):
        # Accuracy
        ax = axes[0][col]
        ax.plot(hd['accuracy'], color=colors['train'], label='train')
        ax.plot(hd['val_accuracy'], color=colors['val'], label='val')
        best_val = max(hd['val_accuracy'])
        best_ep = hd['val_accuracy'].index(best_val)
        ax.axhline(y=best_val, color=colors['val'], linestyle=':', alpha=0.4)
        ax.plot(best_ep, best_val, 'o', color=colors['val'], markersize=6)
        ax.set_title(f'{name}\nbest val: {best_val:.3f} (ep {best_ep+1})', fontsize=11)
        ax.set_xlabel('epoch')
        ax.set_ylim(acc_min, acc_max)
        if col == 0:
            ax.set_ylabel('accuracy')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)

        # Loss
        ax = axes[1][col]
        ax.plot(hd['loss'], color=colors['train'], label='train')
        ax.plot(hd['val_loss'], color=colors['val'], label='val')
        ax.set_title(f'{name} — loss', fontsize=11)
        ax.set_xlabel('epoch')
        ax.set_ylim(0, loss_max)
        if col == 0:
            ax.set_ylabel('loss')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
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


def reset_seed(seed=42):
    """Resetuje wszystkie źródła losowości — Python random, NumPy, TensorFlow.

    Wywołuj na początku każdej komórki, która robi coś losowego (inicjalizacja wag,
    trening z dropoutem, shuffle batchy, sampling z rozkładu) — wtedy wynik jest
    powtarzalny niezależnie od kolejności wykonania komórek.
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as _tf
        _tf.random.set_seed(seed)
    except ImportError:
        pass


def display_generated_city(prompt, real_cities=None):
    """Wyświetla wygenerowaną nazwę miasta z Modelu Języka, ładnie sformatowaną z emoji 🏘️.

    Argument `prompt` to pełny ciąg z tokenami specjalnymi % i ! — np. '%Babciowice!'.
    Jeżeli podasz `real_cities` (set/list nazw treningowych), pod nazwą wyświetli się
    marker czy taka nazwa już istnieje w zbiorze, czy jest nowa.
    """
    from IPython.display import display, Markdown
    name = prompt[1:-1]  # bez tokenów % i !
    msg = f"## 🏘️ **Wygenerowana nazwa miasta:** `{name.title()}`"
    if real_cities is not None:
        if name in real_cities:
            msg += "\n\n⚠️ Ta nazwa **istnieje** w zbiorze treningowym (model ją skopiował)"
        else:
            msg += "\n\n✨ Nowa, **nieistniejąca** nazwa — model ją wymyślił"
    display(Markdown(msg))


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
    # Od najsilniejszego do najsłabszego; MoE przed dense w tej samej rodzinie
    "qwen3.6:35b-a3b", "qwen3.6:27b",
    "qwen3.5:27b", "qwen3.5:9b", "qwen3.5:4b",
    "qwen3:14b", "qwen3:8b",
    "gemma4:12b", "gemma4:4b",
    "gemma4:e4b", "gemma4:e2b",
    "gemma3:27b", "gemma3:12b", "gemma3:8b",
    "qwen2.5:14b", "qwen2.5:7b",
    "gemma2:9b", "mistral:7b", "llama3.1:8b",
    "qwen3.5:2b", "qwen3.5:0.8b",
    "qwen3:4b", "qwen3:1.7b", "qwen3:0.6b",
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


def detect_lmstudio(base_url="http://localhost:1234", api_key=None):
    import requests
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # Natywny endpoint LM Studio
    try:
        r = requests.get(f"{base_url}/api/v1/models", timeout=3, headers=headers)
        if r.status_code == 200:
            models = r.json().get("models", [])
            return [m["key"] for m in models if m.get("type") == "llm"]
    except Exception:
        pass
    # Fallback: standardowy OpenAI endpoint (proxy, vLLM, itp.)
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=3, headers=headers)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                return [m["id"] for m in data]
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
    available_lower = [a.lower() for a in available_models]

    # Przebieg 1: dokładny match z tagiem (Ollama ":" → LM Studio "-")
    for p in preferred:
        p_tag = p.replace(":", "-").lower()  # "qwen3.6:35b-a3b" → "qwen3.6-35b-a3b"
        for i, a in enumerate(available_lower):
            if p_tag in a or p.lower() == a:
                return available_models[i]

    # Przebieg 2: match po rodzinie (fallback gdy brak dokładnego)
    for p in preferred:
        pname = p.split(":")[0].lower()
        for i, a in enumerate(available_lower):
            if pname in a:
                return available_models[i]

    return available_models[0] if available_models else None


def _is_docker():
    """Detect if running inside a Docker container."""
    from pathlib import Path
    return Path("/.dockerenv").exists()


def connect_llm(lecturer_server="http://ADRES_SERWERA:PORT", model=None, api_key=None, backend=None, ports=None):
    """Wykryj działający LLM i zwróć (client, instructor_client, model_name).

    Kolejność prób (gdy backend=None):
      1. LM Studio lokalne (port 1234 + dodatkowe z `ports`)
      2. Auto-launch LM Studio (jeśli `lms` w PATH)
      3. Ollama lokalna (port 11434)
      4. Serwer prowadzącego — próbuje LM Studio, potem Ollama

    Args:
        lecturer_server: adres serwera prowadzącego (fallback gdy brak lokalnego LLM-a)
        model: opcjonalny override — partial-match nazwy modelu (np. "gemma"
               wybierze pierwszy dostępny model z "gemma" w nazwie). Domyślnie
               None = automatyczny wybór najmocniejszego dostępnego.
        api_key: opcjonalny klucz API do serwera LLM (np. LM Studio z włączonym
                 Require Authentication). Domyślnie None = bez klucza.
        backend: wymuszony backend (pomija auto-detekcję). Możliwe wartości:
                 - "lmstudio" — tylko lokalne LM Studio
                 - "ollama" — tylko lokalna Ollama
                 - "lecturer" — tylko serwer prowadzącego (lecturer_server)
                 - lista np. ["lmstudio", "ollama"] — próbuj w podanej kolejności
                 - None — auto-detect: LM Studio → Ollama → serwer (domyślne)
        ports: lista dodatkowych portów do szukania LM Studio na localhost
               (np. [4141, 8080]). Domyślnie None = tylko port 1234.

    Zwraca:
        client           — do function calling (tools=)
        instructor_client — do structured output (response_model=), lub None
        model_name       — nazwa modelu

    Zwraca (None, None, None) jeśli nic nie znalezione.
    """
    from openai import OpenAI

    def _make_clients(base_url, default_key, model):
        """Tworzy oba klienty: zwykły + instructor."""
        key = api_key or default_key  # jawny api_key ma priorytet
        client = OpenAI(base_url=f"{base_url}/v1", api_key=key)
        try:
            import instructor
            instr = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
        except Exception:
            instr = None
        return client, instr, model

    def _pick(models):
        """Wybiera model: najpierw override (partial match), potem auto."""
        if model:
            match = next((m for m in models if model.lower() in m.lower()), None)
            if match:
                return match
            print(f"  ⚠️ Nie znaleziono '{model}' — wybieram automatycznie")
        return pick_best_model(models) or models[0]

    # ── Helpery do prób poszczególnych backendów ─────────────────────
    lms_ports = [1234]
    if ports:
        for p in ports:
            if p not in lms_ports:
                lms_ports.append(p)

    def _try_lmstudio():
        in_docker = _is_docker()
        hosts = ["localhost"]
        if in_docker:
            hosts.append("host.docker.internal")
        for host in hosts:
            for port in lms_ports:
                url = f"http://{host}:{port}"
                label = f"port {port}" if host == "localhost" else f"{host}:{port}"
                print(f"Szukam LM Studio ({label})...")
                models = detect_lmstudio(url, api_key=api_key)
                if not models and host == "localhost" and port == 1234:
                    models = _try_launch_lms() and detect_lmstudio(url, api_key=api_key)
                if models:
                    picked = _pick(models)
                    print(f"✓ LM Studio ({label})! Model: {picked}")
                    return _make_clients(url, "lm-studio", picked)
        return None

    def _try_ollama():
        in_docker = _is_docker()
        hosts = ["localhost"]
        if in_docker:
            hosts.append("host.docker.internal")
        for host in hosts:
            label = "port 11434" if host == "localhost" else f"{host}:11434"
            print(f"Szukam Ollamy ({label})...")
            base = f"http://{host}:11434"
            models = detect_ollama(base)
            if models:
                picked = _pick(models)
                print(f"✓ Ollama ({label})! Model: {picked}")
                return _make_clients(base, "ollama", picked)
        return None

    def _try_lecturer():
        _is_placeholder = not lecturer_server or "ADRES_SERWERA" in lecturer_server
        if not lecturer_server or _is_placeholder:
            print("  Serwer prowadzącego: adres nie ustawiony.")
            return None
        print(f"Próbuję serwer prowadzącego ({lecturer_server})...")
        models = detect_lmstudio(lecturer_server, api_key=api_key)
        if models:
            picked = _pick(models)
            print(f"✓ Serwer prowadzącego! Model: {picked}")
            return _make_clients(lecturer_server, "lm-studio", picked)
        models = detect_ollama(lecturer_server)
        if models:
            picked = _pick(models)
            print(f"✓ Serwer prowadzącego (Ollama)! Model: {picked}")
            return _make_clients(lecturer_server, "ollama", picked)
        return None

    _BACKENDS = {
        "lmstudio": _try_lmstudio,
        "ollama": _try_ollama,
        "lecturer": _try_lecturer,
    }

    # ── Wymuszony backend (string lub lista) ──────────────────────────
    if backend is not None:
        backends = [backend] if isinstance(backend, str) else list(backend)
        print(f"Backend: {', '.join(backends)}")
        for b in backends:
            fn = _BACKENDS.get(b)
            if not fn:
                print(f"  ⚠️ Nieznany backend: '{b}' (dostępne: {', '.join(_BACKENDS)})")
                continue
            result = fn()
            if result:
                return result
            print(f"  ✗ {b} — nie znaleziono")
        return None, None, None

    # ── Auto-detect (backend=None): LM Studio → Ollama → serwer ──────
    result = _try_lmstudio()
    if result:
        return result

    print("  LM Studio niedostępne.")
    result = _try_ollama()
    if result:
        return result

    print("  Lokalny LLM niedostępny.")
    result = _try_lecturer()
    if result:
        return result

    print("✗ Brak dostępnego LLM-a! Zainstaluj LM Studio lub Ollamę (setup_local_llm.ipynb).")
    if _is_docker():
        print("💡 Docker: upewnij się, że LM Studio/Ollama działa na Twoim komputerze (nie w kontenerze).")
        print("   connect_llm automatycznie szuka na host.docker.internal.")
    return None, None, None


def setup_auth_client(client, instructor_client, model_name):
    """Sprawdź czy serwer wymaga auth i poproś o dane studenta.

    Logika:
      - Jeśli client wskazuje na localhost/127.0.0.1 → lokalny LLM, nic nie rób.
      - Jeśli serwer zdalny → sonda BEZ Authorization.
        - 401/403 → pyta o imię + hasło, tworzy nowego klienta z auth.
        - Brak 401 → serwer nie wymaga auth, nic nie rób.

    Zwraca: (client, instructor_client) — oryginalne lub nowe.
    """
    if not client:
        return client, instructor_client

    base_url = str(getattr(client, "base_url", ""))
    is_local = "localhost" in base_url or "127.0.0.1" in base_url
    if is_local:
        return client, instructor_client

    # ── Sonda: czy serwer wymaga hasła? ──
    import requests as _req
    needs_password = False
    try:
        probe = _req.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json={"model": "test", "messages": []},
            timeout=3,
        )
        needs_password = probe.status_code in (401, 403)
    except Exception:
        pass

    # ── Serwer zdalny → zawsze pytaj o imię, hasło tylko gdy wymagane ──
    from openai import OpenAI

    student_name = input("👤 Twoje imię: ").strip() or "Anonim"

    if needs_password:
        import getpass as _gp
        student_key = _gp.getpass("🔑 Hasło (prowadzący poda na zajęciach): ")
    else:
        student_key = None

    new_client = OpenAI(
        base_url=base_url,
        api_key=student_key or "no-key",
        default_headers={"X-Student-Name": student_name},
    )

    # Test połączenia
    try:
        new_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1,
        )
        print(f"✅ Połączono z serwerem jako {student_name}")
    except Exception as e:
        print(f"❌ Błąd połączenia: {e}")
        print("   Sprawdź hasło i spróbuj ponownie.")
        return None, None

    try:
        import instructor
        new_instr = instructor.from_openai(new_client, mode=instructor.Mode.MD_JSON)
    except Exception:
        new_instr = None

    return new_client, new_instr


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


# ── Reasoning helpers (Function Calling) ─────────────────────────────
# Modele LLM mogą zwracać tok myślenia (reasoning) w RÓŻNYCH atrybutach
# w zależności od dostawcy/modelu:
#   - reasoning_content (Qwen3, DeepSeek-R1) — osobny atrybut
#   - reasoning (niektóre OpenAI-compatible)  — osobny atrybut
#   - thought / thinking (inne implementacje) — osobny atrybut
#   - <|channel>thought ... <channel|> (Gemma-4) — w msg.content!

import re as _re

_REASONING_FIELDS = ('reasoning_content', 'reasoning', 'thought', 'thinking')

_CHANNEL_RE = _re.compile(
    r"<\|channel>thought\s*(.*?)\s*<channel\|>",
    _re.DOTALL,
)


def _strip_channel_tokens(text):
    if not text:
        return text, None
    m = _CHANNEL_RE.search(text)
    thinking = m.group(1).strip() if m else None
    cleaned = _CHANNEL_RE.sub("", text).strip()
    return cleaned, thinking


def extract_reasoning(msg):
    """
    Wyciąga natywny tok myślenia z odpowiedzi LLM-a.

    Obsługuje:
      - Qwen3/DeepSeek-R1: atrybut reasoning_content
      - Gemma-4: tokeny <|channel>thought ... <channel|> w msg.content
    """
    attr_reasoning = next(
        (getattr(msg, f, None) for f in _REASONING_FIELDS if getattr(msg, f, None)),
        None
    )
    if attr_reasoning:
        return attr_reasoning
    _, channel_thinking = _strip_channel_tokens(getattr(msg, 'content', None))
    return channel_thinking


def clean_content(msg):
    """
    Zwraca msg.content oczyszczony z artefaktów reasoning (Gemma-4 channel tokens).
    """
    text = getattr(msg, 'content', None)
    cleaned, _ = _strip_channel_tokens(text)
    return cleaned


def print_reasoning(msg, max_chars=500):
    """
    Wyświetla natywny tok myślenia (reasoning) z odpowiedzi LLM-a.
    Obsługuje wszystkie formaty (atrybuty + Gemma-4 channel tokens).
    """
    reasoning = extract_reasoning(msg)
    if reasoning:
        print(f"  🧠 Tok myślenia (reasoning):")
        for line in str(reasoning)[:max_chars].split("\n"):
            print(f"     {line}")
        print()


# ─────────────────────────── Web tools ──────────────────────────────

from html.parser import HTMLParser as _HTMLParser


class _TextExtractor(_HTMLParser):
    """Prosty ekstraktor tekstu z HTML — stdlib, zero zależności."""

    _SKIP_TAGS = frozenset(('script', 'style', 'nav', 'noscript', 'svg'))
    _BLOCK_TAGS = frozenset(('p', 'br', 'div', 'h1', 'h2', 'h3', 'h4',
                             'h5', 'h6', 'li', 'tr', 'article', 'section'))

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag in self._BLOCK_TAGS:
            self._parts.append('\n')

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self):
        text = ''.join(self._parts)
        text = _re.sub(r'[ \t]+', ' ', text)
        text = _re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


def fetch_webpage(url: str, max_chars: int = 3000) -> str:
    """
    Pobiera stronę internetową i zwraca jej treść jako czysty tekst (bez HTML).

    Użyj po search_web, żeby przeczytać pełną treść strony z wyników wyszukiwania.

    Args:
        url: Pełny adres URL strony do pobrania, np. 'https://example.com/article'
        max_chars: Maksymalna liczba znaków do zwrócenia (domyślnie 3000)
    """
    import requests
    try:
        r = requests.get(
            url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; StudentBot/1.0)"},
        )
        r.raise_for_status()
        extractor = _TextExtractor()
        extractor.feed(r.text)
        text = extractor.get_text()
        if not text:
            return f"Strona {url} nie zawiera czytelnego tekstu."
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[...obcięto do {max_chars} znaków]"
        return f"Treść strony {url}:\n\n{text}"
    except Exception as e:
        return f"Nie udało się pobrać {url}: {e}"
