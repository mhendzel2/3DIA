#!/usr/bin/env python3
"""Dose-response fitting regression checks."""

from __future__ import annotations

import sys

sys.path.append("src")


def test_dose_response_fitting() -> None:
    from utils.analysis_utils import fit_dose_response

    concentrations = [0.1, 1.0, 10.0, 100.0, 1000.0]
    responses = [100, 90, 70, 30, 10]

    result = fit_dose_response(concentrations, responses)
    assert "error" not in result, result.get("error", "dose-response fitting failed")
    assert "ic50" in result
    assert "fit_method" in result

    short_conc = [1.0, 10.0]
    short_resp = [90, 30]
    insufficient = fit_dose_response(short_conc, short_resp)
    assert "error" in insufficient
