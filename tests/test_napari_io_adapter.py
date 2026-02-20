"""Tests for napari I/O adapter functions in `pymaris_napari._io`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
