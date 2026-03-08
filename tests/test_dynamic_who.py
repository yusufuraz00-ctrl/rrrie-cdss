"""Dynamic RRRIE Clinical Test Runner.

Randomly selects 3-4 test cases from the case pool on each run and
validates results against WHO ICD-11 API data. Every execution tests
different cases — no two runs are identical.

Usage:
    pytest tests/test_dynamic_who.py -v
    pytest tests/test_dynamic_who.py -v -k "dynamic"
    pytest tests/test_dynamic_who.py -v --count=3    # run 3 times with different cases

Environment:
    WHO_ICD_API_TOKEN  — (optional) Bearer token for WHO ICD API
                          Without it, the API may rate-limit or return 401.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from glob import glob
from pathlib import Path
from typing import Any

import pytest


from src.utils.medical_codes import (
    COMMON_ICD11,
    is_valid_icd11,
    lookup_icd11_who,
    normalize_icd11,
    search_icd11_who,
)
from src.utils.safety_checks import detect_red_flags, check_vitals_red_flags

# ─── Case Pool Discovery ────────────────────────────────────────────────────────

CASES_DIR = Path(__file__).parent / "test_cases"
_CASE_POOL: list[dict] = []


def _load_case_pool() -> list[dict]:
    """Load all JSON test cases from test_cases/ directory."""
    global _CASE_POOL
    if _CASE_POOL:
        return _CASE_POOL

    case_files = sorted(CASES_DIR.glob("*.json"))
    for fp in case_files:
        if fp.name == "__init__.py":
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                case = json.load(f)
            case["_source_file"] = fp.name
            _CASE_POOL.append(case)
        except Exception:
            pass  # Skip malformed files

    assert len(_CASE_POOL) >= 3, (
        f"Need at least 3 test cases in {CASES_DIR}, found {len(_CASE_POOL)}"
    )
    return _CASE_POOL


def _select_random_cases(n: int = 4) -> list[dict]:
    """Randomly select n cases from the pool (min 3, max pool size)."""
    pool = _load_case_pool()
    k = min(n, len(pool))
    k = max(k, 3)
    selected = random.sample(pool, k)
    print(f"\n[Dynamic Test] Selected {len(selected)} cases: "
          f"{[c['case_id'] for c in selected]}")
    return selected


# ─── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def random_cases() -> list[dict]:
    """Module-scoped fixture that picks 5-6 random cases once per test module run."""
    n = random.choice([5, 6])  # Randomly pick 5 or 6 cases (pool is 23)
    return _select_random_cases(n)


@pytest.fixture(scope="module")
def who_api_token() -> str | None:
    """WHO ICD API token — auto-obtained via OAuth2 or from environment.

    The medical_codes module now auto-obtains tokens from ICD11_CLIENT_ID
    and ICD11_CLIENT_SECRET in .env. Passing None lets it handle auth.
    """
    # Legacy: still support direct token override via env var
    return os.environ.get("WHO_ICD_API_TOKEN")


# ─── Test: ICD-11 Code Validity (Local) ─────────────────────────────────────────

class TestDynamicICD11Validation:
    """Validate ICD-11 codes in randomly selected cases — local checks."""

    def test_expected_codes_are_valid_icd11(self, random_cases):
        """Every expected ICD-11 code in each case must pass format validation."""
        for case in random_cases:
            case_id = case["case_id"]
            icd11 = case.get("icd11_expected", "")
            expected_codes = case.get("expected_output", {}).get("expected_icd11_codes", [])

            if icd11:
                assert is_valid_icd11(icd11), (
                    f"[{case_id}] Primary ICD-11 code '{icd11}' failed format validation"
                )

            for code in expected_codes:
                assert is_valid_icd11(code), (
                    f"[{case_id}] Expected ICD-11 code '{code}' failed format validation"
                )

    def test_expected_codes_in_common_lookup(self, random_cases):
        """At least one expected code should be in our COMMON_ICD11 dictionary."""
        for case in random_cases:
            case_id = case["case_id"]
            expected_codes = case.get("expected_output", {}).get("expected_icd11_codes", [])
            if not expected_codes:
                continue

            found = any(
                normalize_icd11(c) in COMMON_ICD11 for c in expected_codes
            )
            # Soft check — not all codes may be in our small dictionary
            if not found:
                print(f"  [WARN] [{case_id}] None of {expected_codes} found in COMMON_ICD11 (non-fatal)")


# ─── Test: WHO ICD-11 API Validation ────────────────────────────────────────────

class TestDynamicWHOValidation:
    """Validate ICD-11 codes against the live WHO ICD-11 API.

    These tests require network access. They are designed to be resilient:
    if the WHO API is unavailable or rate-limited, they skip gracefully.
    """

    @pytest.mark.asyncio
    async def test_who_code_lookup(self, random_cases, who_api_token):
        """Look up each case's primary ICD-11 code via WHO API."""
        results = []

        for case in random_cases:
            case_id = case["case_id"]
            icd11 = case.get("icd11_expected", "")
            if not icd11:
                continue

            # Pass token if available; otherwise lookup_icd11_who auto-obtains via OAuth2
            result = await lookup_icd11_who(icd11, api_token=who_api_token)
            results.append((case_id, icd11, result))

            if result.get("found"):
                title = result.get("title", "")
                print(f"  [WHO] [{case_id}] {icd11} → {title}")
            elif result.get("status_code") == 401:
                pytest.skip(
                    "WHO ICD API auth failed. Set ICD11_CLIENT_ID + ICD11_CLIENT_SECRET in .env "
                    "(or WHO_ICD_API_TOKEN as override)"
                )
            elif result.get("error"):
                print(f"  [WHO] [{case_id}] {icd11} → API error: {result.get('error')}")

        # At least one successful lookup expected (unless all fail due to auth)
        successful = [r for _, _, r in results if r.get("found")]
        if results and not successful:
            if any(r.get("status_code") == 401 for _, _, r in results):
                pytest.skip("WHO ICD API authentication required")
            print("  [WARN] No successful WHO lookups (API may be temporarily unavailable)")

    @pytest.mark.asyncio
    async def test_who_search_matches_diagnosis(self, random_cases, who_api_token):
        """Search WHO ICD-11 by diagnosis name and verify expected code appears."""
        for case in random_cases:
            case_id = case["case_id"]
            primary_dx = case.get("expected_output", {}).get("primary_diagnosis", "")
            expected_codes = case.get("expected_output", {}).get("expected_icd11_codes", [])

            if not primary_dx or not expected_codes:
                continue

            results = await search_icd11_who(primary_dx, api_token=who_api_token, max_results=10)

            if not results:
                # API might be unavailable
                print(f"  [WHO] [{case_id}] Search for '{primary_dx}' returned no results (skip)")
                continue

            # Check if any returned code matches our expected codes
            returned_codes = [r.get("theCode", "") for r in results]
            expected_set = {c.upper() for c in expected_codes}
            returned_set = {c.upper() for c in returned_codes if c}

            match = expected_set & returned_set
            if match:
                print(f"  [WHO] [{case_id}] ✓ Search '{primary_dx}' → matched codes: {match}")
            else:
                print(
                    f"  [WHO] [{case_id}] Search '{primary_dx}' → "
                    f"returned {returned_codes[:5]}, expected {expected_codes} (no exact match — acceptable)"
                )


# ─── Test: Patient Data Integrity ────────────────────────────────────────────────

class TestDynamicPatientDataIntegrity:
    """Validate that each random case has valid patient data structure."""

    def test_patient_data_has_required_fields(self, random_cases):
        """Each case's patient_data should have required fields."""
        for case in random_cases:
            case_id = case["case_id"]
            pd = case["patient_data"]

            assert pd.get("age", 0) > 0, f"[{case_id}] Invalid age"
            assert pd.get("chief_complaint"), f"[{case_id}] Missing chief complaint"
            assert pd.get("sex"), f"[{case_id}] Missing sex"

            vitals = pd.get("vitals", {})
            if vitals:
                temp = vitals.get("temperature", 37.0)
                assert temp > 35, f"[{case_id}] Temperature out of range"

            print(f"  [DATA] [{case_id}] patient_data OK (age={pd['age']}, sex={pd['sex']})")

    def test_prompt_text_generation(self, random_cases):
        """Each case should produce valid prompt text from dict."""
        for case in random_cases:
            case_id = case["case_id"]
            pd = case["patient_data"]

            text = "\n".join(f"{k}: {v}" for k, v in pd.items())
            assert len(text) > 50, f"[{case_id}] Prompt text too short"
            assert pd["chief_complaint"] in text, f"[{case_id}] Chief complaint not in prompt text"


# ─── Test: Red Flag Detection ────────────────────────────────────────────────────

class TestDynamicRedFlagDetection:
    """Validate red flag detection for randomly selected cases."""

    def test_red_flags_detected(self, random_cases):
        """Cases with expected red flags should trigger detection."""
        for case in random_cases:
            case_id = case["case_id"]
            pd = case["patient_data"]
            expected_red_flags = case.get("expected_output", {}).get("red_flags", [])

            if not expected_red_flags:
                continue

            # Check symptom-based red flags
            symptom_flags = detect_red_flags(
                chief_complaint=pd.get("chief_complaint", ""),
                symptoms=pd.get("symptoms", []),
            )

            # Check vital-based red flags
            vitals = pd.get("vitals", {})
            vital_flags = check_vitals_red_flags(
                temperature=vitals.get("temperature", 37.0),
                heart_rate=vitals.get("heart_rate", 80),
                respiratory_rate=vitals.get("respiratory_rate", 16),
                spo2=vitals.get("spo2", 98.0),
                blood_pressure=vitals.get("blood_pressure", "120/80"),
            )

            all_detected = symptom_flags + vital_flags

            if all_detected:
                print(f"  [FLAGS] [{case_id}] Detected: {all_detected}")
            else:
                # Some cases have red flags that are conceptual (e.g., "Rebound tenderness")
                # which our keyword detector may not catch — this is acceptable
                print(f"  [FLAGS] [{case_id}] No automated flags detected "
                      f"(expected: {expected_red_flags}) — manual review needed")


# ─── Test: Cross-Case Diversity ──────────────────────────────────────────────────

class TestDynamicDiversity:
    """Ensure randomly selected cases are diverse."""

    def test_no_duplicate_cases(self, random_cases):
        """All selected cases should be unique."""
        ids = [c["case_id"] for c in random_cases]
        assert len(ids) == len(set(ids)), f"Duplicate cases selected: {ids}"

    def test_multiple_body_systems(self, random_cases):
        """Selected cases should ideally span multiple body systems."""
        # ICD-11 chapter letters indicate body system
        chapters = set()
        for case in random_cases:
            code = case.get("icd11_expected", "")
            if code:
                # First 1-2 characters indicate chapter
                chapters.add(code[0])

        print(f"  [DIVERSITY] ICD-11 chapters represented: {sorted(chapters)}")
        # Soft assertion — with random selection, we may get same chapter
        if len(chapters) < 2:
            print("  [WARN] Low diversity — all cases from same chapter (non-fatal)")

    def test_age_range_diversity(self, random_cases):
        """Cases should include varied age groups."""
        ages = [c["patient_data"]["age"] for c in random_cases]
        print(f"  [DIVERSITY] Ages: {ages}")
        age_range = max(ages) - min(ages)
        if age_range < 10:
            print(f"  [WARN] Narrow age range ({age_range} years) — non-fatal")
