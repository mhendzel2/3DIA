#!/usr/bin/env python3
"""
PyMaris Scientific Image Analyzer - Napari Quick Start
Simplified launcher for the Napari desktop version
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"ERROR: Python 3.8+ required. Current: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_napari_installation():
    """Check if Napari is installed"""
    try:
        import napari
        print(f"✓ Napari {napari.__version__} installed")
        return True
    except ImportError:
        print("✗ Napari not installed")
        return False

def install_napari():
    """Install Napari and dependencies"""
    print("Installing Napari and dependencies...")
    packages = [
        "napari[all]",
        "magicgui",
        "qtpy",
        "numpy",
        "scipy", 
        "scikit-image",
        "matplotlib",
        "tifffile",
        "imageio"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {package}")

def launch_napari():
    """Launch the Napari application"""
    try:
        # Try to run main_napari.py
        if Path("main_napari.py").exists():
            print("Launching PyMaris Napari interface...")
            subprocess.run([sys.executable, "main_napari.py"])
        elif Path("src/main_napari.py").exists():
            print("Launching PyMaris Napari interface...")
            subprocess.run([sys.executable, "src/main_napari.py"])
        else:
            # Fallback to basic napari
            print("Launching basic Napari...")
            import napari
            viewer = napari.Viewer()
            napari.run()
    except Exception as e:
        print(f"Launch failed: {e}")
        print("Try running: python main_napari.py")

def main():
    """Main quick start function"""
    print("PyMaris Scientific Image Analyzer - Napari Quick Start")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        return
    
    # Check Napari installation
    if not check_napari_installation():
        choice = input("Install Napari now? (y/n): ").lower()
        if choice == 'y':
            install_napari()
            if not check_napari_installation():
                print("Installation failed. Please install manually:")
                print("pip install napari[all]")
                input("Press Enter to exit...")
                return
        else:
            print("Please install Napari manually:")
            print("pip install napari[all]")
            input("Press Enter to exit...")
            return
    
    # Launch application
    print("\nStarting Napari...")
    launch_napari()

if __name__ == "__main__":
    main()