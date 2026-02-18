"""Tests for scientific_analyzer tracking API endpoint."""

from __future__ import annotations

import numpy as np
import pytest

import scientific_analyzer as sa


@pytest.mark.skipif(not getattr(sa, "HAS_FLASK", False), reason="Flask not available")
def test_track_endpoint_runs_hungarian_tracking() -> None:
    app = sa.app
    cache = sa.analysis_cache
    session_id = "test-track-session"
    data = np.zeros((3, 16, 16), dtype=float)
    data[0, 4:6, 4:6] = 10.0
    data[1, 5:7, 5:7] = 10.0
    data[2, 6:8, 6:8] = 10.0
    cache.set(
        session_id,
        {
            "original_image": data,
            "processed_images": {},
            "results": {},
        },
    )

    client = app.test_client()
    response = client.post("/track", json={"session_id": session_id, "max_distance": 10.0})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["tracking"]["track_count"] >= 1
