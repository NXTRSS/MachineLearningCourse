import sys
import pkg_resources
import tensorflow as tf

EXPECTED_PACKAGES = {
    "tensorflow": "2.15.0",
    "pillow": "9.4.0",
    "pandas": "1.4.1",
    "scikit-learn": "1.0.2",
    "seaborn": "0.11.2",
    "plotly": "5.1.0",
    "pydot": "1.4.2",
    "graphviz": "2.40.1",
    "jupyterlab": "4.2.5",
    "matplotlib": "3.4.3",
    "ipywidgets": "8.1.2",
}

EXPECTED_PYTHON_VERSION = "3.9.7"

def check_python_version():
    print("\nChecking Python version...\n")
    current_python_version = sys.version.split()[0]
    
    if current_python_version != EXPECTED_PYTHON_VERSION:
        print(f"WARNING: Python Version - Expected {EXPECTED_PYTHON_VERSION}, Found {current_python_version}")
    else:
        print(f"Python Version: OK (Version {current_python_version})")

def check_package_versions():
    print("\nChecking installed packages...\n")
    all_ok = True
    warnings = []
    
    for package, expected_version in EXPECTED_PACKAGES.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            if installed_version != expected_version:
                warnings.append(
                    f"WARNING: Package {package} - Expected {expected_version}, Found {installed_version}"
                )
            else:
                print(f"Package {package}: OK (Version {installed_version})")
        except pkg_resources.DistributionNotFound:
            print(f"ERROR: Package {package} - NOT INSTALLED")
            all_ok = False
    
    for warning in warnings:
        print(warning)
    
    if all_ok and not warnings:
        print("\nEnvironment verification: SUCCESS!")
    elif warnings:
        print("\nEnvironment verification: WARNINGS DETECTED. Review above messages.")
    else:
        print("\nEnvironment verification: FAILED. Please fix the errors above.")

def check_tensorflow_gpu():
    print("\nChecking TensorFlow configuration...\n")
    print(f"TensorFlow Version: {tf.__version__}")
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    print(f"GPU Available: {'YES' if gpu_available else 'NO'}")

if __name__ == "__main__":
    check_python_version()  # Check Python version
    check_package_versions()  # Check package versions
    check_tensorflow_gpu()  # Check TensorFlow configuration
