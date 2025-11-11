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

**Manual Installation:**
```batch
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

See `WINDOWS_INSTALLATION.md` for detailed instructions and troubleshooting.

### Linux/macOS

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python src/main_napari.py
```

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
- Python 3.8+
- 4GB RAM
- 2GB disk space

### Recommended
- Windows 10/11 / Ubuntu 20.04+ / macOS 11+
- Python 3.9+
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
- Ensure Python 3.8+ is installed
- Check "Add Python to PATH" was selected
- Run `install.bat` as Administrator

**Import errors?**
- Reinstall: `rmdir /s /q venv` then `install.bat`
- Update pip: `python -m pip install --upgrade pip`

**Qt errors?**
- Try PyQt5: `pip uninstall PyQt6 && pip install PyQt5`
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
├── requirements.txt         # Python dependencies
├── src/
│   ├── main_napari.py       # Napari interface entry
│   ├── main.py              # Multi-interface launcher
│   └── widgets/             # All analysis widgets
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
