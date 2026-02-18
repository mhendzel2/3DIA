"""Helpers for optional external-tool adapters (license isolation)."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ExternalToolAdapterSpec:
    """Describes invocation details for an external reconstruction adapter."""

    name: str
    command: tuple[str, ...]
    license: str
    environment: dict[str, str] = field(default_factory=dict)
    notes: str = ""


def run_external_tool(
    spec: ExternalToolAdapterSpec,
    *,
    args: list[str] | None = None,
    cwd: str | Path | None = None,
    env_overrides: Mapping[str, str] | None = None,
    timeout_seconds: float = 3600.0,
) -> subprocess.CompletedProcess[str]:
    """Execute an adapter command in a separate process/environment."""
    command = [*spec.command, *(args or [])]
    env = dict(os.environ)
    env.update(spec.environment)
    if env_overrides:
        env.update({str(k): str(v) for k, v in env_overrides.items()})
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
