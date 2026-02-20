#!/usr/bin/env python3
"""
PyMaris Scientific Image Analyzer - Napari Quick Start
Simplified launcher for the Napari desktop version.

This script will install the scientific image analyzer package and then launch Napari.
"""

import sys
import subprocess

def ensure_napari_installed():
    """Check if napari is installed, and if not, install it."""
    try:
        import napari
        print(f"✓ Napari {napari.__version__} is installed.")
        return True
    except ImportError:
        print("Napari not found. Installing napari[all]...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "napari[all]"])
            print("✓ Napari installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Napari installation failed: {e}")
            return False

def install_package():
    """Install the package with napari extras from the repository root."""
    print("Installing PyMaris Scientific Image Analyzer package...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[napari]"])
        print("✓ Package installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation failed: {e}")
        print("Please ensure you have pip installed and that you are running this script from the project root.")
        return False

def launch_napari():
    """Launch the Napari application."""
    print("Launching Napari with PyMaris plugins...")
    # Since the package is now installed, we can just run napari,
    # and the plugins will be discovered automatically.
    import napari
    from qtpy.QtWidgets import QApplication

    # Create a QApplication instance if one doesn't exist.
    # This is necessary to avoid a RuntimeError.
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    napari.run()

def main():
    """Main quick start function"""
    print("PyMaris Scientific Image Analyzer - Napari Quick Start")
    print("=" * 60)
    
    if ensure_napari_installed():
        if install_package():
            launch_napari()
        else:
            input("Press Enter to exit...")
    else:
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
