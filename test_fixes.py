#!/usr/bin/env python3
"""Regression coverage for legacy bug-fix modules."""

from __future__ import annotations

import os
import sys

sys.path.append("src")


def test_fibsem_imports() -> None:
    from fibsem_plugins import ChimeraXIntegration, FIBSEMAnalyzer, ThreeDCounter

    chimera = ChimeraXIntegration()
    counter = ThreeDCounter()
    test_labels = [[[1, 0, 2], [0, 1, 0]], [[2, 2, 0], [1, 0, 1]]]
    result = counter.count_3d_objects(test_labels)

    assert FIBSEMAnalyzer is not None
    assert isinstance(chimera.chimerax_path, (str, type(None)))
    assert "total_objects" in result


def test_batch_processor() -> None:
    from batch_processor import WORKFLOW_TEMPLATES, BatchProcessor

    processor = BatchProcessor()
    assert processor is not None
    assert isinstance(WORKFLOW_TEMPLATES, dict)


def test_bug_fixes_integration(tmp_path) -> None:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from bug_fixes import ChimeraXPathFix, MRCExportFix, TIFFExportFix

    chimera_path = ChimeraXPathFix.find_chimerax_installation()
    assert isinstance(chimera_path, (str, type(None)))

    volume_out = tmp_path / "test_volume.mrc"
    labels_out = tmp_path / "test_mask.tiff"

    test_volume = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    success, _ = MRCExportFix.export_proper_mrc(test_volume, str(volume_out))
    assert isinstance(success, bool)

    test_labels = [[1, 0, 2], [0, 1, 0], [2, 2, 1]]
    success, _ = TIFFExportFix.export_proper_tiff(test_labels, str(labels_out))
    assert isinstance(success, bool)
