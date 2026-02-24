# PyMaris Scientific Image Analyzer



A comprehensive multidimensional image analysis program for fluorescence and electron microscopy, designed to replicate and extend Bitplane Imaris functionality.



## 🚀 Quick Start (Windows)



### One-Click Installation



1. **Install:** Double-click `install.bat`

2. **Run:** Double-click `start.bat`



That's it! The Napari interface will launch with all features ready to use.



### Alternative Quick Start



Simply run `quickstart.bat` - it will install if needed and launch the application.



## 📋 What's Included



### Major Features (Imaris-like)



✅ **Volume Rendering**

- Maximum Intensity Projection (MIP)

- Alpha blending volume rendering

- Orthogonal slice views (XY, XZ, YZ)

- Volume clipping planes



✅ **Filament Tracing**

- Automated neuron/cytoskeleton tracing

- Branch point detection

- Skeleton extraction

- Thickness measurements



✅ **Cell Tracking & Lineage**

- Hungarian algorithm linking

- Gap closing

- Division detection

- Hierarchical lineage trees



✅ **Advanced Segmentation**

- Spot detection (LoG, DoG, DoH)

- Surface rendering (Marching Cubes)

- Watershed segmentation

- AI-based methods (Cellpose, StarDist)



✅ **Statistical Analysis**

- Colocalization analysis

- Object measurements (30+ properties)

- Intensity statistics

- Export to CSV/JSON



✅ **Image Processing**

- Gaussian, median, bilateral filtering

- Multiple thresholding methods

- Morphological operations

- Deconvolution (Richardson-Lucy, Wiener)



## 📦 Installation



### Windows



**Automated Installation (Recommended):**

```batch

# Double-click or run:

install.bat

```

`install.bat` is the single supported installer and includes core dependencies,
format readers, and Imaris/Zarr support.



**Manual Installation:**

```batch

python -m venv venv

venv\Scripts\activate.bat

pip install -r requirements.txt

pip install -e .

```



See `WINDOWS_INSTALLATION.md` for detailed instructions and troubleshooting.



### Linux/macOS


```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core package
pip install -e .

# Install napari plugin dependencies
pip install -e ".[napari]"

# Run application
python src/main_napari.py
```

### Optional extras

```bash
# Web interface
pip install -e ".[web]"

# Advanced I/O and formats
pip install -e ".[io,formats]"

# AI backends
pip install -e ".[ai]"
```

### Headless Core API (Phase 2+)

```python
from pymaris import ImageVolume, ProjectStore, open_image, save_image
from pymaris.backends import DEFAULT_REGISTRY

volume = open_image("sample.tif")
segmentation = DEFAULT_REGISTRY.get_segmentation("watershed").segment_instances(volume)

store = ProjectStore("my_project")
store.save_image_layer("raw", volume)
store.save_label_layer("labels", segmentation.labels)
store.record_workflow_step(
    name="watershed",
    params={},
    inputs=["raw"],
    outputs=["labels"],
)
```

### Headless CLI

```bash
# Inspect image metadata
pymaris-cli open sample.tif --json

# Run workflow JSON on an input image and save to project store
pymaris-cli run-workflow --input sample.tif --workflow workflow.json --project ./my_project

# Re-run workflow from an existing project image layer
pymaris-cli run-project --project ./my_project --workflow workflow.json --image-layer input

# Batch execution (desktop/HPC style)
pymaris-cli run-batch --inputs "data/*.tif" --workflow workflow.json --projects-root ./batch_projects

# Reconstruction plugin execution with QA/model provenance artifacts
pymaris-cli run-reconstruction --input sample.tif --plugin deconvolution --project ./my_project

# Export project outputs
pymaris-cli export --project ./my_project --destination ./export_bundle

# Show traceback details for debugging
pymaris-cli --debug run-workflow --input sample.tif --workflow workflow.json --project ./my_project
```

### Napari Project Store Settings

Project/provenance defaults are user-configurable in napari via the
`Workflow Runner (Core Backends)` widget and `Widget Manager` settings panel:

- `Base Project Dir`
- `Session Naming` (`none`, `fixed`, `timestamp`)
- session name/prefix value
- `Provenance` enable/disable toggle

These values are persisted in `config/widget_config.json` under `project_store`.

CI includes a dedicated headless napari widget smoke test:
`tests/test_napari_widget_smoke.py` (run with `QT_QPA_PLATFORM=offscreen`).

## 🎯 Usage


### Starting the Application



**Windows:**

```batch

start.bat

```



**Linux/macOS:**

```bash

source venv/bin/activate

python src/main_napari.py

```



### Interfaces Available



1. **Napari Desktop** (Primary) - Full-featured GUI

   ```batch

   start.bat

   ```



2. **Enhanced Web Interface**

   ```batch

   venv\Scripts\activate.bat

   python src\main.py enhanced

   ```



3. **Simple Web Interface**

   ```batch

   venv\Scripts\activate.bat

   python src\main.py web

   ```



## 📚 Documentation



- **Features & Usage Guide:** `FEATURES_AND_USAGE.md` - Comprehensive feature documentation
- **Windows Installation:** `WINDOWS_INSTALLATION.md` - Windows-specific setup guide
- **Installation Guide:** `NAPARI_INSTALLATION_GUIDE.md` - General Napari setup
- **Complete Package:** `COMPLETE_PACKAGE_README.md` - Full package information
- **Code Analysis:** `CODE_ANALYSIS_AND_ENHANCEMENTS.md` - Technical details
- **Progress Log:** `PROGRESS.md` - Baseline run status and migration progress
- **Architecture Baseline:** `ARCHITECTURE.md` - Current control flow and coupling map
- **Remake Strategy:** `docs/REMAKE_STRATEGY.md` - Spec-aligned phased rebuild plan
- **Changelog:** `CHANGELOG.md` - Incremental milestone notes
- **Migration Notes:** `MIGRATION.md` - Compatibility and install changes
- **CLI Workflows:** `docs/CLI_WORKFLOWS.md` - Headless workflow and project execution
- **Workflow Schema:** `schemas/workflow.schema.json` - Canonical JSON structure
- **Workflow Examples:** `examples/workflows/*.json` - Ready-to-run templates


## 🔬 Example Workflows



### Workflow 1: 3D Cell Tracking

```python

1. Load 4D image (Time, Z, Y, X)

2. Segmentation Widget → Cellpose/Watershed

3. Cell Tracking Widget → Configure and track

4. View lineage tree

5. Export tracks to CSV

```



### Workflow 2: Neuron Tracing

```python

1. Load neuron image

2. Processing Widget → Enhance contrast

3. Filament Tracing Widget → Auto-trace

4. Analyze branch points

5. Export skeleton and statistics

```



### Workflow 3: Volume Rendering

```python

1. Load 3D confocal stack

2. Volume Rendering Widget → MIP

3. Adjust contrast/brightness

4. Generate publication-quality image

```



## 🛠️ System Requirements



### Minimum
- Windows 10 / Linux / macOS
- Python 3.10+
- 4GB RAM
- 2GB disk space

### Recommended
- Windows 10/11 / Ubuntu 20.04+ / macOS 11+
- Python 3.11+
- 16GB RAM

- 10GB disk space

- Dedicated GPU (future features)



## 📊 Comparison with Imaris



| Feature | PyMaris | Imaris |

|---------|---------|--------|

| Volume Rendering | ✅ | ✅ |

| Cell Tracking | ✅ | ✅ |

| Filament Tracing | ✅ | ✅ |

| Colocalization | ✅ | ✅ |

| Python API | ✅ | Limited |

| Open Source | ✅ | ❌ |

| Cost | **Free** | $$$$$ |



## 🐛 Troubleshooting



**Installation fails?**

- Ensure Python 3.10+ is installed

- Check "Add Python to PATH" was selected

- Run `install.bat` as Administrator



**Import errors?**

- Reinstall: `rmdir /s /q venv` then `install.bat`

- Update pip: `python -m pip install --upgrade pip`



**Qt errors?**

- Reinstall Qt stack: `venv\Scripts\python.exe -m pip install --upgrade --force-reinstall PyQt6 PyQt6-Qt6 qtpy napari`

- Update graphics drivers



**Application is slow?**

- Close unused widgets

- Use Volume Clipping to reduce data

- Increase virtual memory



See `WINDOWS_INSTALLATION.md` for detailed troubleshooting.



## 📁 Project Structure



```
3DIA/
├── install.bat              # Windows installer (run first)
├── start.bat                # Windows launcher
├── quickstart.bat           # Quick launcher
├── pyproject.toml           # Package metadata + extras
├── src/
│   ├── pymaris/             # Headless core library
│   │   ├── io.py            # open_image / save_image
│   │   ├── data_model.py    # ImageVolume canonical model
│   │   ├── project_store.py # provenance + project persistence
│   │   └── backends/        # segmentation/tracking/tracing/restoration adapters
│   ├── pymaris_napari/      # Napari plugin package + NPE2 manifest
│   ├── main_napari.py       # Legacy compatibility launcher shim
│   ├── main.py              # Multi-interface launcher
│   └── widgets/             # Existing GUI widgets
│       ├── volume_rendering_widget.py
│       ├── filament_tracing_widget.py
│       ├── tracking_widget.py
│       └── ... (15+ widgets)
├── config/

│   └── config.json          # Configuration

└── docs/

    ├── FEATURES_AND_USAGE.md

    ├── WINDOWS_INSTALLATION.md

    └── ... (more documentation)

```



## 🤝 Contributing



Contributions welcome! Please:

1. Fork the repository

2. Create a feature branch

3. Make your changes

4. Submit a pull request



## 📄 License



MIT License - Free for academic and commercial use



## 🙏 Acknowledgments



- Inspired by Bitplane Imaris

- Built on Napari platform

- Uses scikit-image, scipy, numpy

- Community contributions welcome



## 📧 Support



- **GitHub Issues:** https://github.com/mhendzel2/3DIA/issues

- **Documentation:** See docs folder

- **Email:** [Your contact info]



## 🎓 Citation



If you use PyMaris in your research:



```bibtex

@software{pymaris2025,

  title = {PyMaris: Open-Source Multidimensional Image Analysis},

  author = {Henderson, Michael},

  year = {2025},

  url = {https://github.com/mhendzel2/3DIA}

}

```



## 🔄 Recent Updates



**v2.1.0** (November 2025)

- ✨ Added Volume Rendering widget with MIP and alpha blending

- ✨ Added Filament Tracing widget for neuron analysis

- ✨ Added Advanced Cell Tracking with lineage trees

- 🐛 Fixed matplotlib backend compatibility

- 🐛 Fixed import errors in timelapse processor

- 📚 Added comprehensive Windows installation scripts



**v2.0.0** (June 2025)

- Initial release with Napari integration

- FIB-SEM specialized tools

- Multi-file timelapse support

- Enhanced web interface



---



**Ready to get started?** Run `install.bat` (Windows) or see installation instructions above!

