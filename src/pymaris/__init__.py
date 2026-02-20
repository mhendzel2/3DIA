"""Headless core package for PyMaris.

The modules in this package intentionally avoid importing napari so they can run
in batch jobs, notebooks, and CLI contexts without a Qt runtime.
"""

from pymaris.analysis import (
    calculate_colocalization_coefficients,
    calculate_distance_measurements,
    calculate_label_distance_measurements,
    calculate_object_statistics,
    generate_euclidean_distance_map,
    load_image,
    segment_cellpose,
    segment_stardist,
    segment_watershed,
)
from pymaris.benchmark import run_baseline_benchmark
from pymaris.data_model import ImageVolume, infer_axes_from_shape, normalize_axes, squeeze_axes
from pymaris.io import list_scenes, open_image, save_image
from pymaris.jobs import JobCancelledError, JobRunner
from pymaris.layers import image_volume_from_layer_data, image_volume_to_layer_data
from pymaris.project_store import ProjectStore
from pymaris.reconstruction import (
    DEFAULT_RECONSTRUCTION_REGISTRY,
    CalibrationArtifact,
    OTFCalibration,
    PSFCalibration,
    ReconstructionEngine,
    ReconstructionPluginInfo,
    ReconstructionRegistry,
    SIMPatternCalibration,
    SMLMPSFModelCalibration,
)
from pymaris.workflow import WorkflowCancelledError, WorkflowResult, WorkflowStep
from pymaris.workflow_runner import ExecutedWorkflowStep, execute_workflow_steps

__all__ = [
    "ImageVolume",
    "JobCancelledError",
    "JobRunner",
    "ProjectStore",
    "ReconstructionEngine",
    "ReconstructionRegistry",
    "ReconstructionPluginInfo",
    "DEFAULT_RECONSTRUCTION_REGISTRY",
    "CalibrationArtifact",
    "PSFCalibration",
    "OTFCalibration",
    "SIMPatternCalibration",
    "SMLMPSFModelCalibration",
    "WorkflowCancelledError",
    "WorkflowResult",
    "WorkflowStep",
    "ExecutedWorkflowStep",
    "load_image",
    "open_image",
    "save_image",
    "list_scenes",
    "segment_cellpose",
    "segment_stardist",
    "segment_watershed",
    "calculate_object_statistics",
    "calculate_colocalization_coefficients",
    "calculate_distance_measurements",
    "generate_euclidean_distance_map",
    "calculate_label_distance_measurements",
    "run_baseline_benchmark",
    "infer_axes_from_shape",
    "normalize_axes",
    "squeeze_axes",
    "image_volume_to_layer_data",
    "image_volume_from_layer_data",
    "execute_workflow_steps",
]
