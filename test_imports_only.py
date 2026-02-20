#!/usr/bin/env python3
"""Basic import/instantiation checks for legacy modules."""

from __future__ import annotations

import os
import sys

sys.path.append("src")


def test_basic_imports() -> None:
    from batch_processor import WORKFLOW_TEMPLATES, BatchProcessor
    from fibsem_plugins import (
        AcceleratedClassification,
        ChimeraXIntegration,
        FIBSEMAnalyzer,
        MembraneSegmenter,
        ThreeDCounter,
        TomoSliceAnalyzer,
    )

    assert ChimeraXIntegration is not None
    assert ThreeDCounter is not None
    assert TomoSliceAnalyzer is not None
    assert AcceleratedClassification is not None
    assert MembraneSegmenter is not None
    assert FIBSEMAnalyzer is not None
    assert BatchProcessor is not None
    assert isinstance(WORKFLOW_TEMPLATES, dict)

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from bug_fixes import ChimeraXPathFix, MRCExportFix, TIFFExportFix

    assert ChimeraXPathFix is not None
    assert MRCExportFix is not None
    assert TIFFExportFix is not None


def test_basic_instantiation() -> None:
    from batch_processor import BatchProcessor
    from fibsem_plugins import ThreeDCounter, TomoSliceAnalyzer

    counter = ThreeDCounter()
    analyzer = TomoSliceAnalyzer()
    processor = BatchProcessor()

    assert counter is not None
    assert analyzer is not None
    assert processor is not None
