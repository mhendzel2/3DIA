"""Tests for canonical image model and core I/O APIs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pymaris.io as pymaris_io
from pymaris.data_model import ImageVolume
from pymaris.io import list_scenes, open_image, save_image
from pymaris.layers import image_volume_from_layer_data, image_volume_to_layer_data


def test_image_volume_scale_for_axes() -> None:
    image = ImageVolume(
        array=np.zeros((2, 3, 4), dtype=np.uint16),
        axes=("Z", "Y", "X"),
        pixel_size={"Z": 2.0, "Y": 0.5, "X": 0.5},
    )
    assert image.scale_for_axes() == [2.0, 0.5, 0.5]


def test_layer_conversion_roundtrip_preserves_axes() -> None:
    image = ImageVolume(
        array=np.ones((4, 8, 8), dtype=np.float32),
        axes=("Z", "Y", "X"),
        metadata={"name": "roundtrip"},
        pixel_size={"Z": 1.5, "Y": 0.25, "X": 0.25},
    )
    layer_data = image_volume_to_layer_data(image=image, name="layer-name")
    restored = image_volume_from_layer_data(data=layer_data[0], metadata=layer_data[1])
    assert restored.axes == ("Z", "Y", "X")
    assert restored.scale_for_axes() == [1.5, 0.25, 0.25]
    np.testing.assert_array_equal(restored.as_numpy(), image.as_numpy())


def test_open_save_image_tiff_roundtrip(tmp_path: Path) -> None:
    source = ImageVolume(
        array=np.arange(27, dtype=np.uint16).reshape(3, 3, 3),
        axes=("Z", "Y", "X"),
        metadata={"name": "synthetic"},
    )
    saved_path = save_image(source, destination=tmp_path / "synthetic.tif", format="tiff")
    assert saved_path.is_file()

    loaded = open_image(saved_path)
    assert loaded.axes == ("Z", "Y", "X")
    np.testing.assert_array_equal(loaded.as_numpy(), source.as_numpy())


def test_image_volume_rejects_mismatched_channel_names() -> None:
    with pytest.raises(ValueError, match="channel_names length"):
        ImageVolume(
            array=np.zeros((3, 16, 16), dtype=np.float32),
            axes=("C", "Y", "X"),
            channel_names=["c0", "c1"],
        )


def test_open_save_image_zarr_roundtrip_preserves_physical_metadata(tmp_path: Path) -> None:
    pytest.importorskip("zarr")

    source = ImageVolume(
        array=np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5),
        axes=("T", "C", "Y", "X"),
        pixel_size={"Y": 0.4, "X": 0.4},
        axis_units={"T": "second", "Y": "micrometer", "X": "micrometer"},
        channel_names=["dna", "actin", "membrane"],
        time_spacing=15.0,
        modality="fluorescence",
        metadata={"name": "synthetic-zarr"},
    )

    saved_path = save_image(source, destination=tmp_path / "synthetic.zarr", format="zarr")
    loaded = open_image(saved_path)

    assert loaded.axes == ("T", "C", "Y", "X")
    assert loaded.pixel_size == {"Y": 0.4, "X": 0.4}
    assert loaded.axis_units["T"] == "second"
    assert loaded.axis_units["Y"] == "micrometer"
    assert loaded.channel_names == ["dna", "actin", "membrane"]
    assert loaded.time_spacing == 15.0
    assert loaded.modality == "fluorescence"
    np.testing.assert_array_equal(loaded.as_numpy(), source.as_numpy())


def test_open_ims_scene_selection_and_scene_listing(tmp_path: Path, monkeypatch) -> None:
    h5py = pytest.importorskip("h5py")
    monkeypatch.setattr(pymaris_io, "HAS_AICSIMAGEIO", False)
    ims_path = tmp_path / "synthetic.ims"
    with h5py.File(ims_path, "w") as handle:
        dataset = handle.create_group("DataSet")
        res0 = dataset.create_group("ResolutionLevel 0")
        tp0 = res0.create_group("TimePoint 0")
        tp1 = res0.create_group("TimePoint 1")
        tp0.create_group("Channel 0").create_dataset(
            "Data",
            data=np.arange(27, dtype=np.uint16).reshape(3, 3, 3),
        )
        tp1.create_group("Channel 0").create_dataset(
            "Data",
            data=np.arange(27, dtype=np.uint16).reshape(3, 3, 3) + 100,
        )

        info = handle.create_group("DataSetInfo")
        image = info.create_group("Image")
        image.attrs["X"] = b"3"
        image.attrs["Y"] = b"3"
        image.attrs["Z"] = b"3"
        image.attrs["ExtMin0"] = b"0"
        image.attrs["ExtMax0"] = b"3"
        image.attrs["ExtMin1"] = b"0"
        image.attrs["ExtMax1"] = b"3"
        image.attrs["ExtMin2"] = b"0"
        image.attrs["ExtMax2"] = b"3"
        image.attrs["Unit"] = b"micrometer"

    scenes = list_scenes(ims_path)
    assert len(scenes) >= 2
    assert scenes[0].startswith("DataSet/ResolutionLevel 0/TimePoint")

    loaded = open_image(ims_path, scene_index=1)
    assert loaded.metadata["scene"].endswith("TimePoint 1")
    assert loaded.axes in {("Z", "Y", "X"), ("C", "Z", "Y", "X")}
    assert loaded.pixel_size.get("X") == 1.0
