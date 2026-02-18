#!/usr/bin/env python3
"""Import checks for Flask batch surface."""

from __future__ import annotations

import sys

sys.path.append("src")


def test_flask_imports() -> None:
    from scientific_analyzer import HAS_BATCH_PROCESSOR, HAS_FLASK

    assert isinstance(HAS_FLASK, bool)
    assert isinstance(HAS_BATCH_PROCESSOR, bool)

    if HAS_BATCH_PROCESSOR:
        from batch_processor import WORKFLOW_TEMPLATES

        assert isinstance(WORKFLOW_TEMPLATES, dict)
        assert WORKFLOW_TEMPLATES
