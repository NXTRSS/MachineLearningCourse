import sys
import subprocess
import shutil
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version

# Minimalne wersje — muszą być spójne z pyproject.toml
required_packages = {
    "tensorflow": "2.20",
    "pillow": "9.4.0",
    "pandas": "2.2.3",
    "scikit-learn": "1.6.0",
    "seaborn": "0.13.0",
    "plotly": "5.1.0",
    "pydot": "2.0",
    "jupyterlab": "4.2.5",
    "matplotlib": "3.9.2",
    "ipywidgets": "8.1.2",
    "gensim": "4.4.0",
    "spacy": "3.8.7",
    "openai": "1.0.0",
}

all_ok = True

print("Checking Python version...\n")
python_version = sys.version_info
if python_version >= (3, 11):
    print(f"Python Version: OK ({'.'.join(map(str, python_version[:3]))})")
else:
    print(f"ERROR: Python {'.'.join(map(str, python_version[:3]))} — wymagany >= 3.11")
    all_ok = False

print("\nChecking installed packages...\n")
for package, min_ver in required_packages.items():
    try:
        installed = version(package)
        if Version(installed) >= Version(min_ver):
            print(f"  {package}: OK ({installed})")
        else:
            print(f"  {package}: WARNING — {installed} < {min_ver} (wymagane >= {min_ver})")
            all_ok = False
    except PackageNotFoundError:
        print(f"  {package}: ERROR — nie zainstalowany!")
        all_ok = False

print("\nChecking system-level packages...\n")

# Sprawdzenie graphviz
if shutil.which("dot"):
    try:
        result = subprocess.run(
            ["dot", "-V"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        graphviz_info = result.stderr.strip() or result.stdout.strip()
        print(f"  graphviz: OK ({graphviz_info})")
    except Exception:
        print("  graphviz: OK (found in PATH)")
else:
    print("  graphviz: NOT FOUND")
    print("    macOS:   brew install graphviz")
    print("    Linux:   sudo apt-get install graphviz")
    print("    Windows: https://graphviz.org/download/")
    all_ok = False

print()
if all_ok:
    print("Environment verification: OK")
else:
    print("Environment verification: PROBLEMS DETECTED")
    print("Sprawdź powyższe błędy i uruchom ponownie.")
