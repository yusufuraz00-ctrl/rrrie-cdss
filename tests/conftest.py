"""Shared pytest configuration and fixtures for RRRIE-CDSS tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── Constants ───────────────────────────────────────────────────────────────────
CASES_DIR = Path(__file__).parent / "test_cases"


# ─── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def all_cases() -> list[dict]:
    """Load ALL test cases (session-scoped — loaded once)."""
    cases = []
    for fp in sorted(CASES_DIR.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            case = json.load(f)
        case["_source_file"] = fp.name
        cases.append(case)
    return cases


@pytest.fixture(scope="session")
def who_token() -> str | None:
    """WHO ICD-11 API bearer token.

    The system auto-obtains tokens via OAuth2 using ICD11_CLIENT_ID +
    ICD11_CLIENT_SECRET from .env. This fixture provides a legacy override.
    """
    return os.environ.get("WHO_ICD_API_TOKEN")


@pytest.fixture(scope="session")
def has_who_token(who_token) -> bool:
    """Whether WHO API token is available."""
    return who_token is not None and len(who_token) > 0


# ─── Markers ─────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "who_api: marks tests that need WHO ICD-11 API token")
    config.addinivalue_line("markers", "integration: marks integration tests that need external services")
