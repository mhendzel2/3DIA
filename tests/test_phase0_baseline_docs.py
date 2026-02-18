"""Phase 0 baseline documentation presence checks."""

from pathlib import Path

REQUIRED_DOCS = (
    "PROGRESS.md",
    "ARCHITECTURE.md",
    "CHANGELOG.md",
)


def test_phase0_docs_exist() -> None:
    missing = [doc for doc in REQUIRED_DOCS if not Path(doc).is_file()]
    assert not missing, f"Missing required baseline docs: {missing}"
