"""Tests for napari widget factory registry wiring."""

from __future__ import annotations

from pymaris_napari._widgets import CONFIG_WIDGET_FACTORIES


def test_workflow_runner_widget_is_registered() -> None:
    assert "workflow_runner" in CONFIG_WIDGET_FACTORIES


def test_distance_tools_widget_is_registered() -> None:
    assert "distance_tools" in CONFIG_WIDGET_FACTORIES
