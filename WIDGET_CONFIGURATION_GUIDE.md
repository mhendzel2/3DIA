# Widget Configuration Guide

## Overview
PyMaris now supports configurable widget loading, allowing you to:
- Choose which widgets load at startup
- Load widgets on-demand during runtime
- Reduce memory usage and startup time
- Customize widget dock areas
- Select pre-configured workspace presets (tracking, HCS, 3D quant)

## Quick Start

### Using Configurable Mode
```batch
run_configurable.bat
```

### Launch Directly Into a Preset
```batch
run_configurable.bat --workspace tracking
run_configurable.bat --workspace high_content_screening
run_configurable.bat --workspace viz_3d_quant
```

List available preset names:
```batch
run_configurable.bat --list-workspaces
```

This launches PyMaris with widget configuration enabled.

## Configuration Methods

### Method 1: Widget Manager GUI (Recommended)
1. Launch PyMaris with `run_configurable.bat`
2. Open the **Widget Manager** dock widget (right side by default)
3. Choose a **Workspace Preset** and click **Apply Preset**
4. Check/uncheck widgets to fine-tune enable/disable settings
5. Select dock area (left/right/top/bottom) for each widget
6. Click **Load Now** to load a widget immediately
7. Click **Save Configuration** to save for next startup
8. Restart PyMaris for changes to take effect

### Method 2: Edit Configuration File
Edit `config/widget_config.json` directly:

```json
{
    "active_workspace": "tracking",
    "workspaces": {
        "default": {
            "enabled_widgets": { "file_io": true, "processing": true, "segmentation": true },
            "widget_areas": { "file_io": "left", "processing": "left", "segmentation": "left" }
        },
        "tracking": {
            "enabled_widgets": { "tracking": true, "distance_tools": true, "hca": false },
            "widget_areas": { "tracking": "right", "distance_tools": "right" }
        }
    },
    "enabled_widgets": {
        "file_io": true,
        "processing": true,
        "segmentation": true,
        "analysis": true,
        "visualization": true,
        "deconvolution": false,
        "statistics": true,
        "filament_tracing": true,
        "tracking": true,
        "simple_threshold": false,
        "adaptive_threshold": false,
        "hca": true
    },
    "widget_areas": {
        "file_io": "left",
        "processing": "left",
        "statistics": "right"
    },
    "load_on_startup": true,
    "show_welcome_message": true
}
```

## Built-in Workspace Presets

These presets are available out of the box in `config/widget_config.json` and from the Widget Manager preset selector:

| Preset | ID | Best For |
|--------|----|----------|
| Default | `default` | Balanced general-purpose microscopy workflow |
| Tracking | `tracking` | Time-lapse tracking and lineage analysis |
| High-Content Screening | `high_content_screening` | Multi-well plate analysis and batch processing |
| 3D Visualization + Quant | `viz_3d_quant` | 3D rendering, quantification, and deconvolution |

You can duplicate any preset under `workspaces` to create your own team-specific layouts.

## Available Widgets

### Essential Widgets (Recommended)
| Widget | ID | Description |
|--------|-----|-------------|
| File I/O | `file_io` | Load/save files (TIFF, CZI, LIF, ND2, Imaris, Zarr) |
| Image Processing | `processing` | Filters, transforms, enhancements |
| Segmentation | `segmentation` | Object detection and cell segmentation |
| Analysis & Plotting | `analysis` | Measurements, colocalization, plotting |
| 3D Visualization | `visualization` | MIP, volume rendering, orthogonal views |
| Statistics | `statistics` | Statistical analysis and data export |

### Advanced Widgets
| Widget | ID | Description |
|--------|-----|-------------|
| Filament Tracing | `filament_tracing` | Neuron/cytoskeleton tracing (Imaris FilamentTracer) |
| Cell Tracking | `tracking` | Cell tracking with lineage trees (Imaris Track) |
| High-Content Analysis | `hca` | Multi-well plate analysis and batch processing |

### Optional Widgets
| Widget | ID | Description |
|--------|-----|-------------|
| Deconvolution | `deconvolution` | Richardson-Lucy and Wiener deconvolution |
| Simple Threshold | `simple_threshold` | Basic threshold widget (magicgui) |
| Adaptive Threshold | `adaptive_threshold` | Adaptive threshold widget (magicgui) |
| AI Segmentation | `ai_segmentation` | Deep learning-based segmentation |
| Biophysics Analysis | `biophysics` | FRAP, FRET, dose-response analysis |
| Interactive Plotting | `interactive_plotting` | Advanced interactive plots |

## Configuration Options

### enabled_widgets
Boolean values controlling which widgets load at startup.
- `true`: Load widget at startup
- `false`: Don't load widget (can load on-demand later)

### widget_areas
Controls where each widget appears in the Napari window.
- `"left"`: Left sidebar
- `"right"`: Right sidebar
- `"top"`: Top area
- `"bottom"`: Bottom area

### load_on_startup
Global toggle for widget loading.
- `true`: Load enabled widgets at startup (default)
- `false`: Start with no widgets, load all on-demand

### show_welcome_message
Controls startup messages.
- `true`: Show welcome and feature list (default)
- `false`: Silent startup

## Performance Tips

### Minimal Configuration (Fastest Startup)
Load only essential widgets:
```json
{
    "enabled_widgets": {
        "file_io": true,
        "processing": true,
        "segmentation": true,
        "analysis": false,
        "visualization": false,
        "statistics": false,
        "filament_tracing": false,
        "tracking": false,
        "hca": false
    }
}
```

### Recommended Configuration (Balanced)
Essential widgets + most-used advanced features:
```json
{
    "enabled_widgets": {
        "file_io": true,
        "processing": true,
        "segmentation": true,
        "analysis": true,
        "visualization": true,
        "statistics": true,
        "filament_tracing": true,
        "tracking": true,
        "hca": false,
        "deconvolution": false
    }
}
```

### Full Configuration (All Features)
All widgets enabled (slower startup, more memory):
```json
{
    "enabled_widgets": {
        "file_io": true,
        "processing": true,
        "segmentation": true,
        "analysis": true,
        "visualization": true,
        "deconvolution": true,
        "statistics": true,
        "filament_tracing": true,
        "tracking": true,
        "hca": true,
        "simple_threshold": true,
        "adaptive_threshold": true
    }
}
```

## On-Demand Widget Loading

Load widgets during runtime without restarting:

1. Open **Widget Manager**
2. Find the widget you want to load
3. Select the dock area (left/right/top/bottom)
4. Click **Load Now**
5. Widget appears immediately

**Note:** On-demand loading doesn't modify the configuration file. To make permanent, also check the widget and click **Save Configuration**.

## Comparison: Standard vs Configurable Mode

| Feature | Standard Mode | Configurable Mode |
|---------|--------------|-------------------|
| Startup Time | ~5-10 seconds | ~2-5 seconds (fewer widgets) |
| Memory Usage | ~500-800 MB | ~200-400 MB (minimal config) |
| Widget Selection | All widgets loaded | Choose which to load |
| On-Demand Loading | No | Yes |
| Configuration File | None | `config/widget_config.json` |
| Launch Command | `run_pymaris.bat` | `run_configurable.bat` |

## Use Cases

### Quick Image Viewing
```json
{
    "enabled_widgets": {
        "file_io": true,
        "visualization": true
    }
}
```
**Benefit:** Fastest startup, minimal memory, perfect for quick checks.

### Analysis Workflow
```json
{
    "enabled_widgets": {
        "file_io": true,
        "processing": true,
        "segmentation": true,
        "analysis": true,
        "statistics": true
    }
}
```
**Benefit:** All analysis tools available, no tracking/tracing overhead.

### Neuron Research
```json
{
    "enabled_widgets": {
        "file_io": true,
        "processing": true,
        "visualization": true,
        "filament_tracing": true
    }
}
```
**Benefit:** Optimized for neuron/filament analysis.

### Cell Tracking Studies
```json
{
    "enabled_widgets": {
        "file_io": true,
        "processing": true,
        "segmentation": true,
        "tracking": true,
        "statistics": true
    }
}
```
**Benefit:** Complete cell tracking workflow.

### High-Content Screening
```json
{
    "enabled_widgets": {
        "file_io": true,
        "processing": true,
        "segmentation": true,
        "hca": true,
        "statistics": true
    }
}
```
**Benefit:** Batch processing of multi-well plates.

## Troubleshooting

### Widget doesn't load
1. Check that the widget is enabled in config: `"widget_name": true`
2. Verify the widget imports correctly (check console for errors)
3. Try loading with **Load Now** button to see error details
4. Check that dependencies are installed (e.g., Flask for HCA widget)

### Configuration file not found
PyMaris creates a default config at `config/widget_config.json` on first run. If missing:
1. Run `run_configurable.bat`
2. Configuration file is created automatically
3. Or manually create the file using the examples above

### Changes don't take effect
1. Save configuration using **Save Configuration** button
2. **Restart PyMaris** (changes apply at startup)
3. Verify `config/widget_config.json` was updated

### Widget Manager not showing
1. Check right sidebar for "Widget Manager" dock
2. If hidden, go to Window → Dock Widgets → Widget Manager
3. Or manually add to `main_napari_configurable.py`

## Migration from Standard Mode

To switch from standard mode to configurable mode:

1. **Backup your work** (config files, custom scripts)
2. Run `run_configurable.bat` instead of `run_pymaris.bat`
3. All widgets load by default on first run
4. Use Widget Manager to customize
5. Your workflow remains the same, just with more control

## Technical Details

### Configuration File Location
- **Path:** `config/widget_config.json`
- **Format:** JSON
- **Encoding:** UTF-8
- **Auto-created:** Yes (on first run if missing)

### Widget Loading Order
Widgets load in this order:
1. Widget Manager (always first)
2. Essential widgets (file_io, processing, segmentation, etc.)
3. Advanced widgets (filament_tracing, tracking, hca)
4. Optional widgets (deconvolution, ai_segmentation, etc.)

### Import Performance
- **Lazy imports:** Widgets imported only when loaded
- **Memory savings:** ~100-400 MB with minimal configuration
- **Startup speed:** 50-70% faster with 5 widgets vs 14 widgets

## Best Practices

1. **Start minimal:** Enable only widgets you use regularly
2. **Use on-demand:** Load specialized widgets when needed
3. **Organize by area:** Group related widgets in same dock area
4. **Save configurations:** Create multiple configs for different workflows
5. **Regular cleanup:** Disable unused widgets to save resources

## See Also
- [NAPARI_FEATURES_OVERVIEW.md](NAPARI_FEATURES_OVERVIEW.md) - Complete widget documentation
- [SUPPORTED_FILE_FORMATS.md](SUPPORTED_FILE_FORMATS.md) - File format support
- [INSTALLATION_TROUBLESHOOTING.md](INSTALLATION_TROUBLESHOOTING.md) - Installation help
