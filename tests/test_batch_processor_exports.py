"""Tests for batch export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from batch_processor import HAS_TIFFFILE, BatchProcessor


def test_export_mask_simple_writes_tiff_or_text(tmp_path: Path) -> None:
    processor = BatchProcessor()
    labels = np.array([[0, 1], [2, 2]], dtype=np.uint16)
    output = tmp_path / "labels.tiff"
    processor._export_mask_simple(labels, str(output))

    if HAS_TIFFFILE:
        assert output.is_file()
    else:
        assert (tmp_path / "labels.txt").is_file()
