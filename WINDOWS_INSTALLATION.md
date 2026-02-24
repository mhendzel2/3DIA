# PyMaris Installation and Startup Guide (Windows)

## Quick Start

### Single Installer Workflow

PyMaris now uses **one installer only**:

```batch
install.bat
```

This script installs:
- Core scientific and Napari dependencies
- Format readers (CZI/LIF/ND2)
- Imaris `.ims` support (`h5py`)
- Zarr support (`zarr`)
- Legacy dependency repair pass behavior integrated into this script

Then launch:

```batch
start.bat
```

Or use auto-install + launch:

```batch
quickstart.bat
```

## Prerequisites

- **Windows 10 or 11**
- **Python 3.10+** from [python.org](https://www.python.org/)
  - Check **Add Python to PATH** during installation
- **8 GB RAM recommended** (4 GB minimum)
- **~3 GB free disk space**

## What install.bat Does

1. Verifies Python is installed and version is 3.10+
2. Creates/reuses `venv`
3. Upgrades `pip`, `setuptools`, `wheel`
4. Installs unified dependencies from `requirements.txt`
5. Runs a dependency repair pass for napari/Qt ecosystem packages
6. Installs local project package (`-e .`)
7. Verifies key imports (including format readers)
8. Runs `pip check` for dependency consistency

## Manual Installation (Alternative)

```batch
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
python src\main_napari.py
```

## Common Startup Commands

**Napari desktop**
```batch
start.bat
```

**Configurable widget mode**
```batch
run_configurable.bat
```

**Web interface**
```batch
venv\Scripts\activate.bat
python src\main.py enhanced
```

## Troubleshooting

Use the dedicated troubleshooting guide for known installation and environment issues:

- `INSTALLATION_TROUBLESHOOTING.md`

## Notes

- This repository intentionally keeps **one supported installer path** for Windows (`install.bat`) to avoid dependency drift.
- If installation fails mid-way, re-run `install.bat` before trying manual package-level fixes.
