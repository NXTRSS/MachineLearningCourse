import sys
import subprocess
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

print("Checking Python version...\n")
python_version = sys.version_info
expected_python_version = (3, 9, 7)
if (python_version.major, python_version.minor, python_version.micro) == expected_python_version:
    print(f"Python Version: OK (Version {'.'.join(map(str, python_version[:3]))})")
else:
    print(f"WARNING: Python Version: {'.'.join(map(str, python_version[:3]))} (Expected: {'.'.join(map(str, expected_python_version))})")

def check_package(package_name, expected_version):
    try:
        installed_version = version(package_name)
        if installed_version == expected_version:
            print(f"Package {package_name}: OK (Version {installed_version})")
        else:
            print(f"WARNING: Package {package_name}: Found Version {installed_version} (Expected: {expected_version})")
    except PackageNotFoundError:
        print(f"ERROR: Package {package_name}: NOT INSTALLED")

def check_system_package(package_name):
    try:
        result = subprocess.run(["conda", "list", package_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and package_name in result.stdout:
            print(f"Package {package_name}: OK (Installed via conda)")
        else:
            print(f"WARNING: Package {package_name}: NOT INSTALLED (System-level package missing)")
    except FileNotFoundError:
        print(f"ERROR: Conda not found. Could not check system-level package {package_name}.")

print("\nChecking installed packages...\n")
for package, expected_version in required_packages.items():
    check_package(package, expected_version)

print("\nChecking system-level packages...\n")
check_system_package("graphviz")

print("\nEnvironment verification: ", end="")
all_ok = True
for package, expected_version in required_packages.items():
    try:
        installed_version = version(package)
        if installed_version != expected_version:
            all_ok = False
    except PackageNotFoundError:
        all_ok = False

if all_ok:
    print("OK")
else:
    print("FAILED. Please fix the errors above.")

