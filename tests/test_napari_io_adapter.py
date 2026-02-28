"""Tests for napari I/O adapter functions in `pymaris_napari._io`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pymaris_napari import _io as io_adapter
from pymaris_napari._io import get_reader, write_tiff


def test_get_reader_for_tiff_and_roundtrip_write(tmp_path: Path) -> None:
    source = tmp_path / "sample.tif"
    data = np.arange(16, dtype=np.uint16).reshape(4, 4)

    from tifffile import imwrite

    imwrite(source, data)

    reader = get_reader(str(source))
    assert reader is not None
    layer_data = reader(str(source))
    assert len(layer_data) == 1

    read_data, metadata, layer_type = layer_data[0]
    assert layer_type == "image"
    np.testing.assert_array_equal(np.asarray(read_data), data)

    destination = tmp_path / "written.tif"
    written = write_tiff(str(destination), read_data, metadata)
    assert Path(written).is_file()


def test_get_reader_for_empty_directory_returns_none_and_warns(monkeypatch, tmp_path: Path) -> None:
    warned: list[str] = []

    monkeypatch.setattr(io_adapter, "_notify_no_readable_files", lambda directory: warned.append(str(directory)))

    reader = get_reader(str(tmp_path))
    assert reader is None
    assert warned == [str(tmp_path)]


def test_directory_reader_prefers_nd_sidecar(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    nd_file = dataset / "sample.nd"
    nd_file.write_text("nd sidecar", encoding="utf-8")
    tif_file = dataset / "sample_t0001.tif"
    tif_file.write_bytes(b"not a real tiff")

    opened_paths: list[str] = []

    class _FakeImage:
        pass

    def _fake_open_image(path: str):
        opened_paths.append(path)
        return _FakeImage()

    def _fake_to_layer_data(image, name: str, layer_type: str):
        return (np.zeros((1, 1), dtype=np.uint8), {"name": name}, layer_type)

    monkeypatch.setattr(io_adapter, "open_image", _fake_open_image)
    monkeypatch.setattr(io_adapter, "image_volume_to_layer_data", _fake_to_layer_data)

    reader = get_reader(str(dataset))
    assert reader is not None
    layer_data = reader(str(dataset))

    assert len(layer_data) == 1
    assert opened_paths == [str(nd_file)]
