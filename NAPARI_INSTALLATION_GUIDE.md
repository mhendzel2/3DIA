# PyMaris Napari Installation Guide

## Overview

This guide covers local installation and launch of the Napari desktop interface for PyMaris.

## Requirements

- Python 3.10+
- Windows 10/11, macOS, or Linux
- 8 GB RAM recommended (4 GB minimum)

## Recommended Installation

### Windows

Use the unified installer:

```batch
install.bat
start.bat
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
python src/main_napari.py
```

## What Gets Installed

`requirements.txt` includes:
- Napari + Qt stack
- Scientific stack (`numpy`, `scipy`, `scikit-image`, `pandas`)
- I/O stack (`tifffile`, `h5py`, `zarr`, `mrcfile`, `imageio`, `pims`)
- Format readers (`czifile`, `readlif`, `nd2reader`)
- Web extras (`Flask`, `Werkzeug`, `requests`)

## Launch Options

### Napari default

```batch
start.bat
```

### Configurable widget mode

```batch
run_configurable.bat
```

### Direct Python launch

```batch
venv\Scripts\python.exe src\main_napari.py
```

## Verify Installation

```batch
test-installation.bat
```

or

```batch
venv\Scripts\python.exe -c "import napari, numpy, scipy, skimage; print('Napari install OK')"
```

## GPU Note

General GPU acceleration is not globally enabled across the full pipeline yet.
Some optional paths (for example, SAM via PyTorch) can use CUDA when available.

## Troubleshooting

If install or startup fails, see:
- `INSTALLATION_TROUBLESHOOTING.md`
