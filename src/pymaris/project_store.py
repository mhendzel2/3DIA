"""Project store and provenance persistence for reproducible workflows."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tifffile

from pymaris.data_model import ImageVolume
from pymaris.io import open_image, save_image
from pymaris.logging import get_logger

LOGGER = get_logger(__name__)

PROVENANCE_SCHEMA_VERSION = "0.2.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    read = 0
    with path.open("rb") as handle:
        while True:
            if read >= max_bytes:
                break
            chunk = handle.read(min(1024 * 1024, max_bytes - read))
            if not chunk:
                break
            hasher.update(chunk)
            read += len(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class WorkflowRecord:
    """Serializable workflow execution record."""

    id: str
    name: str
    params: dict[str, Any]
    inputs: list[str]
    outputs: list[str]
    created_at: str
    workflow_version: str | None = None
    backend: dict[str, Any] | None = None
    result_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "params": self.params,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "created_at": self.created_at,
        }
        if self.workflow_version:
            payload["workflow_version"] = self.workflow_version
        if self.backend:
            payload["backend"] = self.backend
        if self.result_metadata:
            payload["result_metadata"] = self.result_metadata
        return payload


class ProjectStore:
    """Persist inputs, outputs, and provenance for a project directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.inputs_dir = self.root / "inputs"
        self.outputs_dir = self.root / "outputs"
        self.images_dir = self.outputs_dir / "images"
        self.labels_dir = self.outputs_dir / "labels"
        self.tracks_dir = self.outputs_dir / "tracks"
        self.meshes_dir = self.outputs_dir / "meshes"
        self.tables_dir = self.outputs_dir / "tables"
        self.metadata_dir = self.root / "metadata"
        self.provenance_path = self.metadata_dir / "provenance.json"

    def initialize(self) -> None:
        """Create directory layout and initialize provenance if missing."""
        for directory in (
            self.inputs_dir,
            self.images_dir,
            self.labels_dir,
            self.tracks_dir,
            self.meshes_dir,
            self.tables_dir,
            self.metadata_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        if not self.provenance_path.exists():
            provenance = {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "created_at": _utc_now(),
                "inputs": [],
                "outputs": [],
                "workflow_steps": [],
                "environment": self.snapshot_environment(),
            }
            self._write_provenance(provenance)

    def snapshot_environment(self, packages: Sequence[str] | None = None) -> dict[str, Any]:
        """Capture Python and selected package versions."""
        selected = list(
            packages
            or [
                "scientific-image-analyzer",
                "numpy",
                "scipy",
                "scikit-image",
                "tifffile",
                "dask",
                "napari",
            ]
        )
        package_versions: dict[str, str] = {}
        for package_name in selected:
            try:
                package_versions[package_name] = importlib_metadata.version(package_name)
            except importlib_metadata.PackageNotFoundError:
                continue

        return {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions,
            "captured_at": _utc_now(),
        }

    def record_input(self, uri: str | Path) -> dict[str, Any]:
        """Add an input source record to provenance."""
        self.initialize()
        source = Path(uri) if not str(uri).startswith(("http://", "https://")) else None
        entry: dict[str, Any] = {"uri": str(uri), "recorded_at": _utc_now()}
        if source is not None and source.exists():
            entry.update(
                {
                    "path": str(source.resolve()),
                    "size_bytes": int(source.stat().st_size),
                    "sha256_head": _sha256_file(source),
                }
            )
        provenance = self._read_provenance()
        provenance["inputs"].append(entry)
        self._write_provenance(provenance)
        return entry

    def record_workflow_step(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
        workflow_version: str | None = None,
        backend: Mapping[str, Any] | None = None,
        result_metadata: Mapping[str, Any] | None = None,
    ) -> WorkflowRecord:
        """Append a workflow step entry with a stable persisted ID."""
        self.initialize()
        provenance = self._read_provenance()
        next_index = len(provenance["workflow_steps"]) + 1
        record = WorkflowRecord(
            id=f"step-{next_index:04d}",
            name=name,
            params=dict(params or {}),
            inputs=[str(value) for value in (inputs or [])],
            outputs=[str(value) for value in (outputs or [])],
            created_at=_utc_now(),
            workflow_version=str(workflow_version) if workflow_version else None,
            backend=_to_json_compatible(dict(backend or {})) if backend else None,
            result_metadata=_to_json_compatible(dict(result_metadata or {})) if result_metadata else None,
        )
        provenance["workflow_steps"].append(record.to_dict())
        self._write_provenance(provenance)
        return record

    def compare_environment(self, packages: Sequence[str] | None = None) -> dict[str, Any]:
        """Compare current environment with stored provenance environment snapshot."""
        self.initialize()
        provenance = self._read_provenance()
        stored = provenance.get("environment", {})
        if not isinstance(stored, Mapping):
            stored = {}

        stored_packages_raw = stored.get("packages", {})
        stored_packages = (
            {str(key): str(value) for key, value in dict(stored_packages_raw).items()}
            if isinstance(stored_packages_raw, Mapping)
            else {}
        )
        selected_packages = list(packages or sorted(stored_packages.keys()))
        current = self.snapshot_environment(packages=selected_packages if selected_packages else None)
        current_packages_raw = current.get("packages", {})
        current_packages = (
            {str(key): str(value) for key, value in dict(current_packages_raw).items()}
            if isinstance(current_packages_raw, Mapping)
            else {}
        )

        package_mismatches: list[dict[str, Any]] = []
        for package_name in sorted(set(stored_packages.keys()) | set(current_packages.keys())):
            stored_value = stored_packages.get(package_name)
            current_value = current_packages.get(package_name)
            if stored_value != current_value:
                package_mismatches.append(
                    {
                        "package": package_name,
                        "stored": stored_value,
                        "current": current_value,
                    }
                )

        python_stored = str(stored.get("python_version", ""))
        python_current = str(current.get("python_version", ""))
        platform_stored = str(stored.get("platform", ""))
        platform_current = str(current.get("platform", ""))

        return {
            "matches": not package_mismatches
            and python_stored == python_current
            and platform_stored == platform_current,
            "python_version": {"stored": python_stored, "current": python_current},
            "platform": {"stored": platform_stored, "current": platform_current},
            "package_mismatches": package_mismatches,
        }

    def save_image_layer(self, name: str, image: ImageVolume, format: str = "tiff") -> Path:
        """Save an image layer and register it in provenance."""
        self.initialize()
        suffix = "zarr" if format.lower() == "zarr" else "tif"
        destination = self.images_dir / f"{name}.{suffix}"
        saved = save_image(image=image, destination=destination, format=format)
        self._record_output(kind="image", name=name, output_path=saved, metadata=image.metadata_dict())
        return saved

    def load_image_layers(self) -> dict[str, ImageVolume]:
        """Load persisted image layers from outputs/images."""
        self.initialize()
        layers: dict[str, ImageVolume] = {}
        for path in sorted(self.images_dir.glob("*")):
            if path.suffix.lower() not in {".tif", ".tiff", ".zarr"}:
                continue
            layers[path.stem] = open_image(path)
        return layers

    def save_label_layer(self, name: str, labels: Any, format: str = "tiff") -> Path:
        """Save a labels array and register it in provenance."""
        self.initialize()
        if format.lower() not in {"tif", "tiff"}:
            raise ValueError("label saving currently supports only TIFF")
        destination = self.labels_dir / f"{name}.tif"
        labels_array = np.asarray(labels)
        tifffile.imwrite(destination, labels_array, photometric="minisblack")
        self._record_output(
            kind="labels",
            name=name,
            output_path=destination,
            metadata={"dtype": str(labels_array.dtype), "shape": list(labels_array.shape)},
        )
        return destination

    def load_label_layers(self) -> dict[str, np.ndarray]:
        """Load persisted labels arrays from outputs/labels."""
        self.initialize()
        labels: dict[str, np.ndarray] = {}
        for path in sorted(self.labels_dir.glob("*.tif*")):
            labels[path.stem] = np.asarray(tifffile.imread(path))
        return labels

    def save_table(self, name: str, rows: Iterable[Mapping[str, Any]]) -> Path:
        """Save a tabular output as CSV and record it in provenance."""
        self.initialize()
        import csv

        destination = self.tables_dir / f"{name}.csv"
        rows_list = [dict(row) for row in rows]
        fieldnames = sorted({key for row in rows_list for key in row.keys()})
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_list)
        self._record_output(
            kind="table",
            name=name,
            output_path=destination,
            metadata={"rows": len(rows_list), "columns": fieldnames},
        )
        return destination

    def save_tracks(self, name: str, tracks_payload: Mapping[str, Any]) -> Path:
        """Save tracking payload as JSON and register it in provenance."""
        self.initialize()
        destination = self.tracks_dir / f"{name}.json"
        serializable = _to_json_compatible(dict(tracks_payload))
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, sort_keys=True)
        self._record_output(
            kind="tracks",
            name=name,
            output_path=destination,
            metadata={"keys": sorted(serializable.keys())},
        )
        return destination

    def save_graph(self, name: str, graph: Mapping[str, Any]) -> Path:
        """Save trace graph payload as JSON and register it in provenance."""
        self.initialize()
        destination = self.meshes_dir / f"{name}.json"
        serializable = _to_json_compatible(dict(graph))
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, sort_keys=True)
        self._record_output(
            kind="graph",
            name=name,
            output_path=destination,
            metadata={"keys": sorted(serializable.keys())},
        )
        return destination

    def export_outputs(self, destination: str | Path) -> Path:
        """Copy outputs directory to destination and return copied path."""
        self.initialize()
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        export_root = target / "outputs"
        if export_root.exists():
            shutil.rmtree(export_root)
        shutil.copytree(self.outputs_dir, export_root)
        return export_root

    def load_provenance(self) -> dict[str, Any]:
        """Load provenance metadata from disk."""
        self.initialize()
        return self._read_provenance()

    def _record_output(
        self,
        *,
        kind: str,
        name: str,
        output_path: Path,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {
            "kind": kind,
            "name": name,
            "path": str(output_path.resolve()),
            "created_at": _utc_now(),
            "size_bytes": int(output_path.stat().st_size) if output_path.exists() else None,
            "sha256_head": _sha256_file(output_path) if output_path.exists() else None,
            "metadata": dict(metadata or {}),
        }
        provenance = self._read_provenance()
        provenance["outputs"].append(entry)
        self._write_provenance(provenance)
        return entry

    def _read_provenance(self) -> dict[str, Any]:
        with self.provenance_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_provenance(self, payload: Mapping[str, Any]) -> None:
        with self.provenance_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_compatible(item) for item in value]
    return value
