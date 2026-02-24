# PyMaris Installation Troubleshooting

## First Rule

Use the **single installer** first:

```batch
install.bat
```

Do not use legacy install scripts (they were removed).

## Quick Diagnostics

```batch
venv\Scripts\python.exe --version
venv\Scripts\python.exe -m pip check
test-installation.bat
```

## Common Issues

### 1) Python not found

**Error**: `Python is not installed or not in PATH`

**Fix**:
1. Install Python 3.10+ from [python.org](https://www.python.org/)
2. Enable **Add Python to PATH**
3. Re-open terminal and run `install.bat`

---

### 2) Wrong Python version

**Error**: `Python 3.10+ is required`

**Fix**:
- Install Python 3.10, 3.11, or 3.12
- Ensure `python --version` points to the correct interpreter

---

### 3) Virtual environment issues

**Symptoms**: activation fails or missing modules after install

**Fix**:
```batch
rmdir /s /q venv
install.bat
```

---

### 4) Qt/Napari startup errors

**Error**: Qt platform plugin initialization failures

**Fix**:
```batch
venv\Scripts\activate.bat
python -m pip install --upgrade --force-reinstall PyQt6 PyQt6-Qt6 qtpy napari
set QT_QPA_PLATFORM=windows
python src\main_napari.py
```

Also update GPU drivers (NVIDIA/AMD/Intel).

---

### 5) Dependency conflicts

**Error**: `pip check` reports conflicts

**Fix**:
```batch
venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --prefer-binary -r requirements.txt
python -m pip check
```

---

### 6) PowerShell execution policy blocks activation

**Error**:
`Activate.ps1 cannot be loaded because running scripts is disabled`

**Fix options**:
- Use `venv\Scripts\activate.bat` from Command Prompt
- Or run directly without activation:

```batch
venv\Scripts\python.exe src\main_napari.py
```

---

### 7) Optional heavy dependencies fail

Some optional AI/extended ecosystem packages may fail on constrained systems.

**Recommended path**:
1. Keep baseline installation from `install.bat`
2. Start app with `start.bat`
3. Add optional packages later if needed

## Format Support Verification

Check that core readers are present:

```batch
venv\Scripts\python.exe -c "import h5py,zarr,readlif,nd2reader,czifile,pims,tifffile; print('Format readers OK')"
```

## If Problems Persist

1. Capture environment details:
```batch
python --version
venv\Scripts\python.exe -m pip --version
venv\Scripts\python.exe -m pip list > installed_packages.txt
```
2. Save full console output from `install.bat`
3. Open an issue with the logs and your Windows/Python versions
