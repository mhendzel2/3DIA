"""NPE2 manifest structure tests."""

from __future__ import annotations

from importlib.resources import files

import tomllib
import yaml


def test_manifest_has_commands_readers_writers_widgets() -> None:
    manifest_text = files("pymaris_napari").joinpath("napari.yaml").read_text(encoding="utf-8")
    manifest = yaml.safe_load(manifest_text)

    contributions = manifest["contributions"]
    assert "commands" in contributions
    assert "readers" in contributions
    assert "writers" in contributions
    assert "widgets" in contributions


def test_manifest_contains_required_command_ids() -> None:
    manifest_text = files("pymaris_napari").joinpath("napari.yaml").read_text(encoding="utf-8")
    manifest = yaml.safe_load(manifest_text)

    command_ids = {entry["id"] for entry in manifest["contributions"]["commands"]}

    required = {
        "scientific-image-analyzer.get_reader",
        "scientific-image-analyzer.write_tiff",
        "scientific-image-analyzer.widget.file_io",
        "scientific-image-analyzer.widget.processing",
        "scientific-image-analyzer.widget.segmentation",
        "scientific-image-analyzer.widget.analysis",
        "scientific-image-analyzer.widget.distance_tools",
        "scientific-image-analyzer.widget.workflow_runner",
    }
    assert required.issubset(command_ids)


def test_manifest_reader_patterns_include_zarr() -> None:
    manifest_text = files("pymaris_napari").joinpath("napari.yaml").read_text(encoding="utf-8")
    manifest = yaml.safe_load(manifest_text)
    readers = manifest["contributions"]["readers"]
    patterns = set(readers[0]["filename_patterns"])
    assert "*.zarr" in patterns
    assert "*.ome.zarr" in patterns
    assert readers[0]["accepts_directories"] is True


def test_pyproject_points_to_new_manifest() -> None:
    with open("pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)

    entrypoint = pyproject["project"]["entry-points"]["napari.manifest"][
        "scientific-image-analyzer"
    ]
    assert entrypoint == "pymaris_napari:napari.yaml"


def test_pyproject_registers_cli_script() -> None:
    with open("pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)
    scripts = pyproject["project"]["scripts"]
    assert scripts["pymaris-cli"] == "pymaris.cli:main"
