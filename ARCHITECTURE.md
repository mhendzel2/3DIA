# ARCHITECTURE

## Scope
This document is a factual baseline of the current repository state before Phase 1 refactoring.

## Entrypoints and control flow

### Primary launcher
- `src/main.py:10` defines `main()` and dispatches runtime mode from CLI args.
- `src/main.py:23` -> `launch_napari_interface()`.
- `src/main.py:25` -> `launch_web_interface()`.
- `src/main.py:27` -> `launch_enhanced_web_interface()`.
- Default behavior is enhanced web fallback path (`src/main.py:33`, `src/main.py:34`).

### Napari desktop entrypoints
- `src/main_napari.py:20` creates a `napari.Viewer` and directly docks widgets.
- Widget docking is explicit and static (`src/main_napari.py:26`, `src/main_napari.py:31`, `src/main_napari.py:36`, `src/main_napari.py:68`, `src/main_napari.py:74`).
- Event loop starts at `src/main_napari.py:100`.

- `src/main_napari_configurable.py:165` is a configurable variant.
- It loads JSON config from `Path(__file__).parent / "config" / "widget_config.json"` (`src/main_napari_configurable.py:51`) and conditionally imports widgets in `load_widget()` (`src/main_napari_configurable.py:67`).
- Event loop starts at `src/main_napari_configurable.py:206`.

### Web entrypoints
- Enhanced Flask app is built in `src/scientific_analyzer.py` when Flask is available (`src/scientific_analyzer.py:163`, `src/scientific_analyzer.py:164`).
- Core routes:
  - upload: `src/scientific_analyzer.py:183`
  - segmentation: `src/scientific_analyzer.py:238`
  - analysis: `src/scientific_analyzer.py:311`
  - tracking endpoint (not implemented): `src/scientific_analyzer.py:348`
  - batch endpoints: `src/scientific_analyzer.py:430`, `src/scientific_analyzer.py:461`, `src/scientific_analyzer.py:472`
- Fallback HTTP server is `src/simple_analyzer.py` (`src/simple_analyzer.py:208`).

## Where UI widgets live and how they call algorithms

### Widget locations
- Napari widgets are in `src/widgets/`.
- Launcher imports are direct module imports, not plugin command indirection (`src/main_napari.py:8` to `src/main_napari.py:18`).

### Algorithm invocation patterns
- **Segmentation widget**
  - Uses `SegmentationThread` (`src/widgets/segmentation_widget.py:23`) and calls skimage/scipy directly (`src/widgets/segmentation_widget.py:15`, `src/widgets/segmentation_widget.py:17`).
  - UI starts thread via `start_segmentation()` (`src/widgets/segmentation_widget.py:565`).
  - Statistics call utility directly (`src/widgets/segmentation_widget.py:621`).

- **AI segmentation widget**
  - `AISegmentationThread` wraps direct calls to Cellpose/StarDist/SAM (`src/widgets/ai_segmentation_widget.py:54`, `src/widgets/ai_segmentation_widget.py:143`, `src/widgets/ai_segmentation_widget.py:163`, `src/widgets/ai_segmentation_widget.py:184`).
  - Model checkpoint download is performed in widget module (`src/widgets/ai_segmentation_widget.py:34`, `src/widgets/ai_segmentation_widget.py:43`).

- **Tracking widget**
  - `TrackingThread` implements linking internally using Hungarian assignment (`src/widgets/tracking_widget.py:24`, `src/widgets/tracking_widget.py:101`, `src/widgets/tracking_widget.py:144`).
  - Widget converts tracks into napari `Tracks` layer format itself (`src/widgets/tracking_widget.py:579`).

- **Filament tracing widget**
  - `FilamentTracingThread` performs filtering, thresholding, skeletonization, and branch analysis in-thread (`src/widgets/filament_tracing_widget.py:24`, `src/widgets/filament_tracing_widget.py:46`, `src/widgets/filament_tracing_widget.py:68`, `src/widgets/filament_tracing_widget.py:100`).

- **Processing widget**
  - Mixed model: some ops run synchronously in widget handlers (`src/widgets/processing_widget.py:403`, `src/widgets/processing_widget.py:434`, `src/widgets/processing_widget.py:461`), while deconvolution uses a worker thread (`src/widgets/processing_widget.py:28`, `src/widgets/processing_widget.py:824`).

- **Deconvolution widget**
  - Dedicated thread (`src/widgets/deconvolution_widget.py:14`) with Richardson-Lucy/Wiener (`src/widgets/deconvolution_widget.py:40`, `src/widgets/deconvolution_widget.py:53`).

- **Analysis/statistics widgets**
  - Colocalization uses a worker thread and utility functions (`src/widgets/analysis_widget.py:31`, `src/widgets/analysis_widget.py:28`, `src/widgets/analysis_widget.py:472`).
  - Statistics widget computes regionprops directly in UI handler (`src/widgets/statistics_widget.py:114`, `src/widgets/statistics_widget.py:127`).

## I/O locations and current format support

### Napari reader/writer plugin module
- `src/file_io_napari.py:30` reader hook `get_reader` currently matches: `.czi`, `.lif`, `.nd2`, `.oib`, `.tif`, `.tiff`.
- `src/file_io_napari.py:66` AICSImageIO path uses `get_image_dask_data("TCZYX", S=0)`.
- `src/file_io_napari.py:92` tifffile path reads eagerly with `tifffile.imread`.
- `src/file_io_napari.py:140` writer exports TIFF only.

### File I/O widget
- `src/widgets/file_io_widget.py:93` `SmartFileLoader` routes by extension/available library.
- Supported paths include TIFF/CZI/LIF/ND2/MRC/Zarr/Imaris/HDF5/common image formats (`src/widgets/file_io_widget.py:103`, `src/widgets/file_io_widget.py:118`, `src/widgets/file_io_widget.py:121`, `src/widgets/file_io_widget.py:297`).
- Most loaders are eager (`src/widgets/file_io_widget.py:234`, `src/widgets/file_io_widget.py:271`, `src/widgets/file_io_widget.py:273`, `src/widgets/file_io_widget.py:326`).
- Series loading stacks all frames into RAM (`src/widgets/file_io_widget.py:652`, `src/widgets/file_io_widget.py:653`, `src/widgets/file_io_widget.py:670`).

### Web upload/read path
- Upload route stores file and calls `au.load_image(...)` (`src/scientific_analyzer.py:195`, `src/scientific_analyzer.py:199`).
- `au.load_image` uses PIL grayscale conversion (`src/utils/analysis_utils.py:37`, `src/utils/analysis_utils.py:43`).

## Feature location map

### Segmentation
- Classical: `src/widgets/segmentation_widget.py`.
- AI: `src/widgets/ai_segmentation_widget.py`.
- Utility segmentation functions: `src/utils/analysis_utils.py:48`, `src/utils/analysis_utils.py:65`, `src/utils/analysis_utils.py:83`.

### Tracking
- Napari widget and algorithm in same file: `src/widgets/tracking_widget.py`.
- Web endpoint explicitly returns "not implemented": `src/scientific_analyzer.py:348`, `src/scientific_analyzer.py:360`.

### Tracing
- Filament tracing in `src/widgets/filament_tracing_widget.py`.

### Restoration / denoising
- Deconvolution widget: `src/widgets/deconvolution_widget.py`.
- Processing widget deconvolution/filters: `src/widgets/processing_widget.py:279`, `src/widgets/processing_widget.py:807`.
- Additional denoising algorithms in `src/advanced_analysis.py:311`.

### Metrics/statistics
- Colocalization + result reporting: `src/widgets/analysis_widget.py`.
- Regionprops table + plotting: `src/widgets/statistics_widget.py`.
- Shared statistics utilities: `src/utils/analysis_utils.py:99`, `src/utils/analysis_utils.py:338`, `src/utils/analysis_utils.py:499`.

### Export paths
- Segmentation CSV export: `src/widgets/segmentation_widget.py:711`.
- Tracking CSV/JSON export: `src/widgets/tracking_widget.py:607`, `src/widgets/tracking_widget.py:646`.
- Filament TIFF/CSV export: `src/widgets/filament_tracing_widget.py:453`, `src/widgets/filament_tracing_widget.py:478`.
- Analysis JSON/CSV/PDF export: `src/widgets/analysis_widget.py:571`, `src/widgets/analysis_widget.py:610`.
- HCA exports: `src/widgets/hca_widget.py:368`, `src/widgets/hca_widget.py:386`.
- Batch exports: `src/batch_processor.py:470`, `src/batch_processor.py:517`, `src/batch_processor.py:547`.

## Packaging and plugin wiring (current)
- Build metadata is in `pyproject.toml` (`pyproject.toml:1`, `pyproject.toml:5`).
- Napari manifest entry point is declared as `scientific_analyzer:napari.yaml` (`pyproject.toml:82`, `pyproject.toml:83`).
- Manifest file is `src/napari.yaml`.
- Manifest contributes readers/writers/widgets (`src/napari.yaml:8`), but it does not define a `commands:` section and references widget command IDs that are not valid python-callable command targets (`src/napari.yaml:29`, `src/napari.yaml:31`, `src/napari.yaml:33`, `src/napari.yaml:35`).

## Coupling and architectural issues (baseline)

### Core/UI coupling
- Algorithm code is embedded directly inside napari widget modules (e.g., tracking, filament, segmentation) rather than isolated in a napari-free core.
- Utility module imports napari types directly (`src/utils/image_utils.py:10`, `src/utils/image_utils.py:11`), so even helper code is not UI-independent.

### Web/desktop duplication
- Desktop widgets and web routes each invoke segmentation/analysis logic through separate paths:
  - desktop: `src/widgets/segmentation_widget.py`, `src/widgets/analysis_widget.py`
  - web: `src/scientific_analyzer.py:238`, `src/scientific_analyzer.py:311`
- Shared behavior exists in `utils.analysis_utils`, but orchestration and outputs differ between UI surfaces.

### Configuration/path fragility
- Config path in configurable napari launcher points to `src/config/widget_config.json` (`src/main_napari_configurable.py:51`), while repository config currently lives at top-level `config/widget_config.json`.
- Widget manager imports `load_widget` via `from src.main_napari_configurable import load_widget` (`src/widgets/widget_manager.py:280`), creating runtime path assumptions.

### Incomplete background-job consistency
- Some long-running operations use `QThread` (segmentation/tracking/filament/deconvolution/HCA/file loading/AI).
- Several heavy processing operations still run synchronously in button handlers (`src/widgets/processing_widget.py:403`, `src/widgets/processing_widget.py:461`, `src/widgets/statistics_widget.py:114`).
- Cancellation exists in only some paths (`src/widgets/segmentation_widget.py:36`, `src/widgets/ai_segmentation_widget.py:66`), and not consistently wired across all workers.

### Data model and provenance gaps
- No canonical image object spanning web + napari + batch paths; each module passes raw numpy arrays and ad-hoc dict metadata.
- No project store/provenance structure for reproducibility; workflow/batch status is in-memory dictionaries (`src/batch_processor.py:58`, `src/batch_processor.py:66`, `src/scientific_analyzer.py:172`).

### Logging/observability
- Core and widget modules primarily use `print(...)` for status/errors (examples: `src/main.py:12`, `src/file_io_napari.py:63`, `src/widgets/segmentation_widget.py:633`, `src/scientific_analyzer.py:22`) instead of structured logging.

## Threading summary (baseline)
- `QThread` present in:
  - `src/widgets/segmentation_widget.py:23`
  - `src/widgets/tracking_widget.py:24`
  - `src/widgets/filament_tracing_widget.py:24`
  - `src/widgets/deconvolution_widget.py:14`
  - `src/widgets/ai_segmentation_widget.py:54`
  - `src/widgets/hca_widget.py:33`
  - `src/widgets/file_io_widget.py:465`
- Additional Python thread pool/background worker in web batch engine:
  - `src/batch_processor.py:117`
  - `src/batch_processor.py:130`
