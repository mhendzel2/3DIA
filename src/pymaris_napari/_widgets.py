"""Widget adapters for NPE2 and legacy launchers."""

from __future__ import annotations

from typing import Any, Callable

import napari.viewer

WidgetFactory = Callable[[Any], Any]


def file_io_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.file_io_widget import FileIOWidget

    return FileIOWidget(napari_viewer)


def processing_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.processing_widget import ProcessingWidget

    return ProcessingWidget(napari_viewer)


def segmentation_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.segmentation_widget import SegmentationWidget

    return SegmentationWidget(napari_viewer)


def analysis_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.analysis_widget import AnalysisWidget

    return AnalysisWidget(napari_viewer)


def visualization_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.visualization_widget import VisualizationWidget

    return VisualizationWidget(napari_viewer)


def deconvolution_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.deconvolution_widget import DeconvolutionWidget

    return DeconvolutionWidget(napari_viewer)


def statistics_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.statistics_widget import StatisticsWidget

    return StatisticsWidget(napari_viewer)


def filament_tracing_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.filament_tracing_widget import FilamentTracingWidget

    return FilamentTracingWidget(napari_viewer)


def tracking_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.tracking_widget import AdvancedTrackingWidget

    return AdvancedTrackingWidget(napari_viewer)


def ai_segmentation_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.ai_segmentation_widget import AISegmentationWidget

    return AISegmentationWidget(napari_viewer)


def hca_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.hca_widget import HighContentAnalysisWidget

    return HighContentAnalysisWidget(napari_viewer)


def biophysics_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.biophysics_widget import BiophysicsWidget

    return BiophysicsWidget(napari_viewer)


def interactive_plotting_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.interactive_plotting_widget import InteractivePlottingWidget

    return InteractivePlottingWidget(napari_viewer)


def distance_tools_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.distance_transform_widget import DistanceTransformWidget

    return DistanceTransformWidget(napari_viewer)


def simple_threshold_widget(_: napari.viewer.Viewer) -> Any:
    from widgets.magicgui_analysis_widget import simple_threshold_widget as widget

    return widget


def adaptive_threshold_widget(_: napari.viewer.Viewer) -> Any:
    from widgets.magicgui_analysis_widget import adaptive_threshold_widget as widget

    return widget


def widget_manager_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from widgets.widget_manager import WidgetManagerWidget

    return WidgetManagerWidget(napari_viewer)


def workflow_runner_widget(napari_viewer: napari.viewer.Viewer) -> Any:
    from pymaris_napari.workflow_widget import WorkflowRunnerWidget

    return WorkflowRunnerWidget(napari_viewer)


CONFIG_WIDGET_FACTORIES: dict[str, WidgetFactory] = {
    "file_io": file_io_widget,
    "processing": processing_widget,
    "segmentation": segmentation_widget,
    "analysis": analysis_widget,
    "visualization": visualization_widget,
    "deconvolution": deconvolution_widget,
    "statistics": statistics_widget,
    "filament_tracing": filament_tracing_widget,
    "tracking": tracking_widget,
    "simple_threshold": simple_threshold_widget,
    "adaptive_threshold": adaptive_threshold_widget,
    "hca": hca_widget,
    "ai_segmentation": ai_segmentation_widget,
    "biophysics": biophysics_widget,
    "interactive_plotting": interactive_plotting_widget,
    "distance_tools": distance_tools_widget,
    "workflow_runner": workflow_runner_widget,
}
