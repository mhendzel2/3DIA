"""Tests for benchmark harness utilities."""

from __future__ import annotations

from pymaris.benchmark import run_baseline_benchmark


def test_run_baseline_benchmark_returns_summary() -> None:
    report = run_baseline_benchmark(repeats=1, size_2d=24, size_3d=(8, 16, 16), seed=1)
    assert report["suite"] == "baseline"
    assert report["summary"]["case_count"] >= 3
    assert report["summary"]["failed_cases"] == 0
    assert len(report["cases"]) == report["summary"]["case_count"]
