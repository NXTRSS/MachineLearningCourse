import sys
import subprocess
import shutil
from importlib.metadata import version, PackageNotFoundError

required_packages = {
    "tensorflow": "2.15.0",
    "pillow": "9.4.0",
    "pandas": "1.4.1",
    "scikit-learn": "1.0.2",
    "seaborn": "0.11.2",
    "plotly": "5.1.0",
    "pydot": "1.4.2",
    "jupyterlab": "4.2.5",
    "matplotlib": "3.4.3",
    "ipywidgets": "8.1.2",
}

all_ok = True

print("Checking Python version...\n")
python_version = sys.version_info
expected_python_version = (3, 9)
if (python_version.major, python_version.minor) == expected_python_version:
    print(f"Python Version: OK (Version {'.'.join(map(str, python_version[:3]))})")
else:
    print(f"WARNING: Python Version: {'.'.join(map(str, python_version[:3]))} (Expected: 3.9.x)")
    all_ok = False

print("\nChecking installed packages...\n")
for package, expected_version in required_packages.items():
    try:
        installed_version = version(package)
        if installed_version == expected_version:
            print(f"Package {package}: OK (Version {installed_version})")
        else:
            print(f"WARNING: Package {package}: Found Version {installed_version} (Expected: {expected_version})")
    except PackageNotFoundError:
        print(f"ERROR: Package {package}: NOT INSTALLED")
        all_ok = False

print("\nChecking system-level packages...\n")

# Sprawdzenie graphviz — działa niezależnie od conda/docker/uv
if shutil.which("dot"):
    try:
        result = subprocess.run(
            ["dot", "-V"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        graphviz_info = result.stderr.strip() or result.stdout.strip()
        print(f"Package graphviz: OK ({graphviz_info})")
    except Exception:
        print("Package graphviz: OK (found in PATH)")
else:
    print("WARNING: Package graphviz: NOT FOUND in PATH")
    print("  -> Docker/uv: graphviz powinien być zainstalowany automatycznie")
    print("  -> Conda: conda install graphviz")
    print("  -> Linux: sudo apt-get install graphviz")
    print("  -> Mac: brew install graphviz")
    print("  -> Windows: https://graphviz.org/download/")

print("\nEnvironment verification: ", end="")
if all_ok:
    print("OK")
else:
    print("WARNINGS DETECTED.")
    print("\nSome packages have mismatched versions or are missing.")
    print("While the environment may be functional, it is recommended")
    print("to align package versions to ensure consistency.")
