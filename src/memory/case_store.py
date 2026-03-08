"""3-Tier Memory System — Human-like experience accumulation.

Architecture (mirrors how experienced doctors learn):

  Tier 1 — RAW CASE BUFFER
    Last ~50 cases with full details. FTS5-searchable for similar cases.
    Oldest cases are distilled into patterns, then removed.

  Tier 2 — PATTERNS
    Distilled lessons from multiple Tier 1 cases. Each pattern tracks:
    - What triggers it (keywords, symptom combinations)
    - What the lesson is (what IE kept finding wrong)
    - How many cases support it (confidence grows with evidence)
    - How many cases contradict it (confidence drops)
    Patterns are NEVER deleted — only their confidence changes.

  Tier 3 — CORE PRINCIPLES
    Promoted from Tier 2 when support_count >= PROMOTION_THRESHOLD.
    These are always injected into every prompt — they represent
    the system's most reliable learned wisdom.

Key invariant: NO EXPERIENCE IS EVER LOST.
  Raw cases may be purged from Tier 1, but their lessons live on in Tier 2/3.

Consolidation triggers:
  - Tier 1 buffer >= 75% capacity
  - Same error type repeated 3+ times in recent cases
  - Large disagreement between R3 confidence and IE confidence
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("rrrie-cdss.memory")

# ── Configuration ───────────────────────────────────────────────────
TIER1_CAPACITY = 50          # Max raw cases before consolidation pressure
CONSOLIDATION_BATCH = 20     # How many old cases to distill at once
PROMOTION_THRESHOLD = 5      # support_count needed for Tier2→Tier3
PATTERN_MIN_SUPPORT = 2      # Min cases to form a pattern
MAX_PRINCIPLES = 30          # Cap Tier 3 to keep prompt injection small
MAX_PATTERNS = 200           # Cap Tier 2
SIMILAR_CASE_LIMIT = 2       # How many similar cases to retrieve
RELEVANT_PATTERN_LIMIT = 5   # How many patterns to inject

# ── Data directory ──────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DATA_DIR / "memory.db"

# ── Common medical stopwords to skip in keyword extraction ──────────
_MEDICAL_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
    "been", "be", "will", "would", "could", "should", "may", "might",
    "do", "does", "did", "not", "no", "yes", "and", "or", "but", "in",
    "on", "at", "to", "for", "of", "with", "by", "from", "as", "if",
    "that", "this", "these", "those", "it", "its", "he", "she", "his",
    "her", "they", "their", "we", "our", "i", "my", "me", "you", "your",
    "patient", "presented", "presents", "reports", "history", "year",
    "years", "old", "male", "female", "man", "woman", "ago", "day",
    "days", "week", "weeks", "month", "months", "also", "which", "who",
    "when", "what", "how", "where", "than", "then", "there", "here",
    "been", "being", "having", "about", "after", "before", "during",
    "between", "into", "through", "over", "under", "up", "down",
    "out", "off", "some", "any", "all", "each", "every", "both",
    "few", "more", "most", "other", "such", "only", "very", "can",
    "just", "so", "too", "now", "new", "well", "time", "noted",
    "revealed", "showed", "shows", "found", "given", "taken", "per",
    # Turkish stopwords
    "bir", "bir", "ile", "için", "olan", "olarak", "gibi", "daha",
    "çok", "var", "yok", "ama", "veya", "hem", "kadar", "sonra",
    "önce", "arasında", "üzerinde", "altında", "içinde", "dışında",
    "hasta", "hastanın", "başvurdu", "şikayeti", "yaşında", "kadın",
    "erkek", "gün", "hafta", "aydır", "yıldır", "önce", "beri",
    "olan", "eden", "etmiş", "olmuş", "olan", "etti", "oldu",
})


@dataclass
class MemoryContext:
    """Structured memory retrieval result for prompt injection."""
    principles: list[str] = field(default_factory=list)      # Tier 3
    patterns: list[dict] = field(default_factory=list)        # Tier 2 (relevant)
    similar_cases: list[dict] = field(default_factory=list)   # Tier 1 (FTS5 match)

    @property
    def is_empty(self) -> bool:
        return not self.principles and not self.patterns and not self.similar_cases

    def format_for_prompt(self) -> str:
        """Format memory into a prompt-ready string (~300-500 tokens max)."""
        if self.is_empty:
            return ""

        sections = []

        # Tier 3: Core Principles (always injected)
        if self.principles:
            lines = [f"- {p}" for p in self.principles]
            sections.append(
                "## Clinical Experience — Core Principles\n"
                "These are validated lessons from past case evaluations:\n"
                + "\n".join(lines)
            )

        # Tier 2: Relevant Patterns
        if self.patterns:
            lines = []
            for p in self.patterns[:RELEVANT_PATTERN_LIMIT]:
                conf = p.get("confidence", 0.5)
                support = p.get("support_count", 0)
                lines.append(
                    f"- [{conf:.0%} confidence, {support} cases] {p['lesson']}"
                )
            sections.append(
                "## Clinical Experience — Relevant Patterns\n"
                + "\n".join(lines)
            )

        # Tier 1: Similar Past Cases
        if self.similar_cases:
            for i, c in enumerate(self.similar_cases[:SIMILAR_CASE_LIMIT], 1):
                dx = c.get("primary_diagnosis", "Unknown")
                ie_conf = c.get("ie_confidence", "?")
                ie_dec = c.get("ie_decision", "?")
                issues = c.get("ie_issues_summary", "None recorded")
                iters = c.get("iteration_count", 1)
                sections.append(
                    f"## Similar Past Case #{i}\n"
                    f"- Diagnosis: {dx}\n"
                    f"- IE confidence: {ie_conf}, decision: {ie_dec}\n"
                    f"- Iterations needed: {iters}\n"
                    f"- Key issues found: {issues}"
                )

        return "\n\n".join(sections)


class CaseStore:
    """SQLite-backed 3-tier memory with FTS5 similarity search.

    Thread-safe via SQLite's internal locking. Single writer, multiple readers.
    All methods are synchronous (SQLite ops are sub-ms on local disk).
    """

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        logger.info("[MEMORY] Initialized. DB: %s", self.db_path)

    def _init_schema(self):
        """Create tables if they don't exist."""
        self.conn.executescript("""
            -- Tier 1: Raw case buffer
            CREATE TABLE IF NOT EXISTS tier1_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_text TEXT NOT NULL,
                patient_keywords TEXT DEFAULT '',
                primary_diagnosis TEXT DEFAULT '',
                r3_confidence REAL DEFAULT 0.0,
                ie_decision TEXT DEFAULT 'FINALIZE',
                ie_confidence REAL DEFAULT 0.0,
                ie_issues TEXT DEFAULT '[]',
                ie_issues_summary TEXT DEFAULT '',
                iteration_count INTEGER DEFAULT 1,
                r1_model TEXT DEFAULT '',
                r3_model TEXT DEFAULT '',
                created_at REAL DEFAULT 0.0
            );

            -- FTS5 for similarity search on Tier 1
            CREATE VIRTUAL TABLE IF NOT EXISTS tier1_fts USING fts5(
                patient_text,
                primary_diagnosis,
                ie_issues_summary,
                content='tier1_cases',
                content_rowid='id'
            );

            -- Triggers to keep FTS5 in sync with tier1_cases
            CREATE TRIGGER IF NOT EXISTS tier1_ai AFTER INSERT ON tier1_cases BEGIN
                INSERT INTO tier1_fts(rowid, patient_text, primary_diagnosis, ie_issues_summary)
                VALUES (new.id, new.patient_text, new.primary_diagnosis, new.ie_issues_summary);
            END;

            CREATE TRIGGER IF NOT EXISTS tier1_ad AFTER DELETE ON tier1_cases BEGIN
                INSERT INTO tier1_fts(tier1_fts, rowid, patient_text, primary_diagnosis, ie_issues_summary)
                VALUES ('delete', old.id, old.patient_text, old.primary_diagnosis, old.ie_issues_summary);
            END;

            CREATE TRIGGER IF NOT EXISTS tier1_au AFTER UPDATE ON tier1_cases BEGIN
                INSERT INTO tier1_fts(tier1_fts, rowid, patient_text, primary_diagnosis, ie_issues_summary)
                VALUES ('delete', old.id, old.patient_text, old.primary_diagnosis, old.ie_issues_summary);
                INSERT INTO tier1_fts(rowid, patient_text, primary_diagnosis, ie_issues_summary)
                VALUES (new.id, new.patient_text, new.primary_diagnosis, new.ie_issues_summary);
            END;

            -- Tier 2: Patterns (distilled from Tier 1)
            CREATE TABLE IF NOT EXISTS tier2_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                trigger_keywords TEXT DEFAULT '[]',
                lesson TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                support_count INTEGER DEFAULT 1,
                contradiction_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.5,
                source_case_ids TEXT DEFAULT '[]',
                first_seen REAL DEFAULT 0.0,
                last_reinforced REAL DEFAULT 0.0
            );

            -- Tier 3: Core Principles (promoted from Tier 2)
            CREATE TABLE IF NOT EXISTS tier3_principles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                principle TEXT NOT NULL UNIQUE,
                category TEXT DEFAULT 'general',
                support_count INTEGER DEFAULT 5,
                contradiction_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.9,
                promoted_from_pattern_id INTEGER,
                first_seen REAL DEFAULT 0.0,
                last_reinforced REAL DEFAULT 0.0,
                FOREIGN KEY (promoted_from_pattern_id) REFERENCES tier2_patterns(id)
            );

            -- Consolidation log
            CREATE TABLE IF NOT EXISTS consolidation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cases_processed INTEGER DEFAULT 0,
                patterns_created INTEGER DEFAULT 0,
                patterns_reinforced INTEGER DEFAULT 0,
                principles_promoted INTEGER DEFAULT 0,
                timestamp REAL DEFAULT 0.0
            );
        """)
        self.conn.commit()

    # ════════════════════════════════════════════════════════════════
    # STORE — Save a completed case
    # ════════════════════════════════════════════════════════════════

    def store_case(
        self,
        patient_text: str,
        primary_diagnosis: str,
        r3_confidence: float,
        ie_decision: str,
        ie_confidence: float,
        ie_issues: list[dict[str, Any]],
        iteration_count: int = 1,
        r1_model: str = "",
        r3_model: str = "",
    ) -> int:
        """Store a completed case in Tier 1. Returns the case ID."""
        keywords = " ".join(self._extract_keywords(patient_text))
        issues_json = json.dumps(ie_issues, ensure_ascii=False)
        issues_summary = "; ".join(
            f"{iss.get('type', '?')}: {iss.get('detail', '?')[:80]}"
            for iss in ie_issues[:5]
        ) if ie_issues else "No issues"

        cursor = self.conn.execute(
            """INSERT INTO tier1_cases
               (patient_text, patient_keywords, primary_diagnosis,
                r3_confidence, ie_decision, ie_confidence,
                ie_issues, ie_issues_summary, iteration_count,
                r1_model, r3_model, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (patient_text, keywords, primary_diagnosis,
             r3_confidence, ie_decision, ie_confidence,
             issues_json, issues_summary, iteration_count,
             r1_model, r3_model, time.time()),
        )
        self.conn.commit()
        case_id = cursor.lastrowid
        logger.info(
            "[MEMORY] Stored case #%d: %s (IE: %s, conf=%.2f, iters=%d)",
            case_id, primary_diagnosis, ie_decision, ie_confidence, iteration_count,
        )
        return case_id

    # ════════════════════════════════════════════════════════════════
    # RETRIEVE — Get relevant memory for a new case
    # ════════════════════════════════════════════════════════════════

    def retrieve_context(self, patient_text: str) -> MemoryContext:
        """Retrieve all relevant memory for a new case.

        Returns:
            MemoryContext with principles (Tier 3), relevant patterns (Tier 2),
            and similar past cases (Tier 1).

        Total injection target: ~300-500 tokens.
        """
        ctx = MemoryContext()

        # Tier 3: ALL core principles (always injected)
        rows = self.conn.execute(
            "SELECT principle FROM tier3_principles "
            "WHERE confidence >= 0.6 "
            "ORDER BY confidence DESC, support_count DESC "
            "LIMIT ?",
            (MAX_PRINCIPLES,),
        ).fetchall()
        ctx.principles = [r["principle"] for r in rows]

        # Tier 2: Patterns matching this case's keywords
        case_keywords = self._extract_keywords(patient_text)
        if case_keywords:
            ctx.patterns = self._find_relevant_patterns(case_keywords)

        # Tier 1: Similar past cases via FTS5
        ctx.similar_cases = self._find_similar_cases(patient_text, case_keywords)

        return ctx

    def _find_relevant_patterns(self, case_keywords: list[str]) -> list[dict]:
        """Find Tier 2 patterns whose triggers overlap with case keywords."""
        all_patterns = self.conn.execute(
            "SELECT * FROM tier2_patterns "
            "WHERE confidence >= 0.4 "
            "ORDER BY confidence DESC, support_count DESC",
        ).fetchall()

        scored = []
        case_kw_set = set(kw.lower() for kw in case_keywords)

        for pat in all_patterns:
            try:
                trigger_kw = set(json.loads(pat["trigger_keywords"]))
            except (json.JSONDecodeError, TypeError):
                trigger_kw = set()

            if not trigger_kw:
                continue

            overlap = len(case_kw_set & trigger_kw)
            if overlap > 0:
                score = overlap * pat["confidence"] * pat["support_count"]
                scored.append((score, dict(pat)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:RELEVANT_PATTERN_LIMIT]]

    def _find_similar_cases(
        self, patient_text: str, keywords: list[str]
    ) -> list[dict]:
        """Find similar past cases using FTS5 full-text search."""
        if not keywords:
            return []

        # Build FTS5 query: use top medical terms joined with OR
        search_terms = keywords[:10]
        # Escape FTS5 special chars
        safe_terms = []
        for t in search_terms:
            t_clean = re.sub(r'[^\w]', '', t)
            if t_clean and len(t_clean) >= 3:
                safe_terms.append(f'"{t_clean}"')
        if not safe_terms:
            return []

        fts_query = " OR ".join(safe_terms)

        try:
            rows = self.conn.execute(
                "SELECT t.*, rank FROM tier1_fts f "
                "JOIN tier1_cases t ON t.id = f.rowid "
                "WHERE tier1_fts MATCH ? "
                "ORDER BY rank "
                "LIMIT ?",
                (fts_query, SIMILAR_CASE_LIMIT),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError as exc:
            logger.warning("[MEMORY] FTS5 search failed: %s", exc)
            return []

    # ════════════════════════════════════════════════════════════════
    # CONSOLIDATION — Tier1 → Tier2 → Tier3
    # ════════════════════════════════════════════════════════════════

    def should_consolidate(self) -> bool:
        """Determine if consolidation should run.

        Triggers:
          1. Tier 1 buffer >= 75% capacity
          2. Same error type repeated 3+ times in last 10 cases
          3. High R3↔IE confidence disagreement in recent cases
        """
        tier1_count = self._tier1_count()

        # Trigger 1: Buffer pressure
        if tier1_count >= TIER1_CAPACITY * 0.75:
            return True

        # Trigger 2: Repeated error type (learning urgency)
        if tier1_count >= 5:
            recent = self.conn.execute(
                "SELECT ie_issues FROM tier1_cases "
                "ORDER BY created_at DESC LIMIT 10",
            ).fetchall()
            error_counter: Counter = Counter()
            for row in recent:
                try:
                    issues = json.loads(row["ie_issues"])
                    for iss in issues:
                        error_counter[iss.get("type", "unknown")] += 1
                except (json.JSONDecodeError, TypeError):
                    pass
            if any(count >= 3 for count in error_counter.values()):
                return True

        # Trigger 3: Confidence disagreement (interesting cases)
        if tier1_count >= 5:
            recent = self.conn.execute(
                "SELECT r3_confidence, ie_confidence FROM tier1_cases "
                "ORDER BY created_at DESC LIMIT 10",
            ).fetchall()
            disagreements = sum(
                1 for r in recent
                if abs(r["r3_confidence"] - r["ie_confidence"]) > 0.3
            )
            if disagreements >= 3:
                return True

        return False

    def consolidate(self) -> dict[str, int]:
        """Run consolidation: distill oldest Tier 1 cases into Tier 2 patterns.

        Process:
          1. Take oldest CONSOLIDATION_BATCH cases from Tier 1
          2. Group by IE issue types
          3. Extract patterns (frequency-based, no LLM needed)
          4. Create/reinforce Tier 2 patterns
          5. Promote Tier 2 → Tier 3 where support is high enough
          6. Delete consolidated cases from Tier 1 (lessons preserved in Tier 2)

        Returns:
            Stats dict with counts of created/reinforced/promoted.
        """
        tier1_count = self._tier1_count()
        if tier1_count < CONSOLIDATION_BATCH:
            return {"cases_processed": 0, "patterns_created": 0,
                    "patterns_reinforced": 0, "principles_promoted": 0}

        # Step 1: Get oldest batch
        oldest = self.conn.execute(
            "SELECT * FROM tier1_cases ORDER BY created_at ASC LIMIT ?",
            (CONSOLIDATION_BATCH,),
        ).fetchall()
        oldest = [dict(r) for r in oldest]

        if not oldest:
            return {"cases_processed": 0, "patterns_created": 0,
                    "patterns_reinforced": 0, "principles_promoted": 0}

        # Step 2: Extract patterns
        new_patterns = self._extract_patterns(oldest)

        # Step 3: Create or reinforce Tier 2 patterns
        created = 0
        reinforced = 0

        for pat in new_patterns:
            existing = self._find_matching_pattern(pat["category"], pat["trigger_keywords"])

            if existing:
                # Reinforce existing pattern
                new_support = existing["support_count"] + pat["support_count"]
                old_sources = json.loads(existing["source_case_ids"]) if existing["source_case_ids"] else []
                new_sources = old_sources + pat["source_case_ids"]
                # Keep last 50 source IDs to avoid unbounded growth
                new_sources = new_sources[-50:]
                new_confidence = new_support / (new_support + existing["contradiction_count"])

                self.conn.execute(
                    """UPDATE tier2_patterns
                       SET support_count = ?,
                           confidence = ?,
                           source_case_ids = ?,
                           last_reinforced = ?,
                           description = ?,
                           lesson = ?
                       WHERE id = ?""",
                    (new_support, new_confidence,
                     json.dumps(new_sources), time.time(),
                     pat["description"], pat["lesson"],
                     existing["id"]),
                )
                reinforced += 1
            else:
                # Create new pattern
                self.conn.execute(
                    """INSERT INTO tier2_patterns
                       (description, trigger_keywords, lesson, category,
                        support_count, confidence, source_case_ids,
                        first_seen, last_reinforced)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (pat["description"],
                     json.dumps(pat["trigger_keywords"]),
                     pat["lesson"],
                     pat["category"],
                     pat["support_count"],
                     pat.get("confidence", 0.5),
                     json.dumps(pat["source_case_ids"]),
                     time.time(), time.time()),
                )
                created += 1

        # Step 4: Promote strong patterns to Tier 3
        promoted = self._promote_patterns()

        # Step 5: Delete consolidated cases from Tier 1
        case_ids = [c["id"] for c in oldest]
        placeholders = ",".join("?" * len(case_ids))
        self.conn.execute(
            f"DELETE FROM tier1_cases WHERE id IN ({placeholders})",
            case_ids,
        )

        # Step 6: Log consolidation
        self.conn.execute(
            """INSERT INTO consolidation_log
               (cases_processed, patterns_created, patterns_reinforced,
                principles_promoted, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (len(oldest), created, reinforced, promoted, time.time()),
        )

        self.conn.commit()

        stats = {
            "cases_processed": len(oldest),
            "patterns_created": created,
            "patterns_reinforced": reinforced,
            "principles_promoted": promoted,
        }
        logger.info("[MEMORY] Consolidation complete: %s", stats)
        return stats

    def _extract_patterns(self, cases: list[dict]) -> list[dict]:
        """Extract patterns from a batch of cases using frequency analysis.

        Groups cases by IE issue type, finds common keywords across cases
        with the same issue, and generates pattern descriptions.

        No LLM needed — pure statistical extraction.
        """
        # Group all issues across cases
        issue_groups: dict[str, list[dict]] = defaultdict(list)

        for case in cases:
            try:
                issues = json.loads(case.get("ie_issues", "[]"))
            except (json.JSONDecodeError, TypeError):
                issues = []

            for issue in issues:
                issue_type = issue.get("type", "unknown")
                issue_groups[issue_type].append({
                    "detail": issue.get("detail", ""),
                    "severity": issue.get("severity", "minor"),
                    "case_id": case["id"],
                    "diagnosis": case.get("primary_diagnosis", ""),
                    "keywords": self._extract_keywords(case.get("patient_text", "")),
                })

        patterns = []
        for issue_type, instances in issue_groups.items():
            if len(instances) < PATTERN_MIN_SUPPORT:
                continue

            # Find recurring diagnosis contexts
            dx_counter: Counter = Counter(
                i["diagnosis"] for i in instances if i["diagnosis"]
            )
            top_dx = dx_counter.most_common(3)

            # Find common keywords across all cases with this issue
            kw_counter: Counter = Counter()
            for inst in instances:
                kw_counter.update(inst["keywords"])
            # Keywords that appear in at least 2 instances
            common_kw = [kw for kw, cnt in kw_counter.most_common(10) if cnt >= 2]

            # Find common severity
            sev_counter = Counter(i["severity"] for i in instances)
            dominant_severity = sev_counter.most_common(1)[0][0]

            # Build human-readable description and lesson
            dx_str = ", ".join(f"{dx} ({cnt}x)" for dx, cnt in top_dx) if top_dx else "various diagnoses"
            detail_samples = [i["detail"][:100] for i in instances[:3]]

            description = (
                f"'{issue_type}' issue detected {len(instances)} times "
                f"(severity: {dominant_severity}). "
                f"Common diagnoses: {dx_str}. "
                f"Samples: {'; '.join(detail_samples)}"
            )

            # The lesson is the actionable takeaway
            if issue_type == "hallucination":
                lesson = (
                    f"When evaluating cases involving {', '.join(common_kw[:3]) or dx_str}, "
                    f"verify ALL reported findings against original patient text. "
                    f"Hallucination detected in {len(instances)} past cases."
                )
            elif issue_type == "missing_history":
                lesson = (
                    f"Cases involving {', '.join(common_kw[:3]) or dx_str} "
                    f"frequently have incomplete history extraction. "
                    f"Pay extra attention to procedures, timelines, and medications."
                )
            elif issue_type in ("unexplained_symptom", "symptom_not_covered"):
                lesson = (
                    f"Primary diagnoses in cases with {', '.join(common_kw[:3]) or 'these symptoms'} "
                    f"often fail to explain all symptoms. Consider broader differentials."
                )
            elif issue_type == "evidence_gap":
                lesson = (
                    f"Cases involving {', '.join(common_kw[:3]) or dx_str} "
                    f"often lack PubMed evidence support. Confidence should be adjusted downward."
                )
            else:
                lesson = (
                    f"Recurring '{issue_type}' pattern in cases with "
                    f"{', '.join(common_kw[:3]) or dx_str}. "
                    f"Flagged {len(instances)} times — requires attention."
                )

            patterns.append({
                "description": description,
                "trigger_keywords": common_kw,
                "lesson": lesson,
                "category": issue_type,
                "support_count": len(instances),
                "confidence": len(instances) / (len(instances) + 1),  # Laplace smoothing
                "source_case_ids": list(set(i["case_id"] for i in instances)),
            })

        # Also extract diagnosis-level patterns (disagreement patterns)
        # Cases where R3 was confident but IE said ITERATE
        disagree_cases = [
            c for c in cases
            if c.get("ie_decision") == "ITERATE"
            and c.get("r3_confidence", 0) >= 0.7
        ]
        if len(disagree_cases) >= PATTERN_MIN_SUPPORT:
            dx_counter = Counter(c["primary_diagnosis"] for c in disagree_cases if c["primary_diagnosis"])
            common_kw_all: Counter = Counter()
            for c in disagree_cases:
                common_kw_all.update(self._extract_keywords(c.get("patient_text", "")))
            top_kw = [kw for kw, _ in common_kw_all.most_common(8) if _ >= 2]

            patterns.append({
                "description": (
                    f"R3 high confidence but IE disagreed in {len(disagree_cases)} cases. "
                    f"Diagnoses: {', '.join(dx for dx, _ in dx_counter.most_common(3))}"
                ),
                "trigger_keywords": top_kw,
                "lesson": (
                    f"Be cautious with high-confidence diagnoses in cases involving "
                    f"{', '.join(top_kw[:3]) or 'similar presentations'}. "
                    f"IE found issues {len(disagree_cases)} times despite R3 confidence."
                ),
                "category": "confidence_disagreement",
                "support_count": len(disagree_cases),
                "confidence": len(disagree_cases) / (len(disagree_cases) + 1),
                "source_case_ids": [c["id"] for c in disagree_cases],
            })

        return patterns

    def _find_matching_pattern(
        self, category: str, trigger_keywords: list[str]
    ) -> dict | None:
        """Find an existing Tier 2 pattern that matches category + keywords."""
        candidates = self.conn.execute(
            "SELECT * FROM tier2_patterns WHERE category = ?",
            (category,),
        ).fetchall()

        new_kw_set = set(kw.lower() for kw in trigger_keywords)
        best_match = None
        best_overlap = 0

        for row in candidates:
            try:
                existing_kw = set(json.loads(row["trigger_keywords"]))
            except (json.JSONDecodeError, TypeError):
                existing_kw = set()

            overlap = len(new_kw_set & existing_kw)
            # Match if >50% keyword overlap (relative to smaller set)
            min_size = min(len(new_kw_set), len(existing_kw)) if existing_kw else 1
            if min_size > 0 and overlap / min_size >= 0.5 and overlap > best_overlap:
                best_overlap = overlap
                best_match = dict(row)

        return best_match

    def _promote_patterns(self) -> int:
        """Promote Tier 2 patterns with high support to Tier 3 principles."""
        candidates = self.conn.execute(
            """SELECT * FROM tier2_patterns
               WHERE support_count >= ?
               AND confidence >= 0.7
               AND id NOT IN (
                   SELECT promoted_from_pattern_id FROM tier3_principles
                   WHERE promoted_from_pattern_id IS NOT NULL
               )
               ORDER BY confidence DESC, support_count DESC""",
            (PROMOTION_THRESHOLD,),
        ).fetchall()

        promoted = 0
        current_count = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM tier3_principles"
        ).fetchone()["cnt"]

        for pat in candidates:
            if current_count + promoted >= MAX_PRINCIPLES:
                break

            # Check if a very similar principle already exists
            existing = self.conn.execute(
                "SELECT id FROM tier3_principles WHERE category = ?",
                (pat["category"],),
            ).fetchone()

            if existing:
                # Reinforce existing principle instead of creating duplicate
                self.conn.execute(
                    """UPDATE tier3_principles
                       SET support_count = support_count + ?,
                           confidence = MIN(0.99, confidence + 0.02),
                           last_reinforced = ?
                       WHERE id = ?""",
                    (pat["support_count"], time.time(), existing["id"]),
                )
            else:
                # Promote to new principle
                try:
                    self.conn.execute(
                        """INSERT INTO tier3_principles
                           (principle, category, support_count, contradiction_count,
                            confidence, promoted_from_pattern_id, first_seen, last_reinforced)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (pat["lesson"], pat["category"],
                         pat["support_count"], pat["contradiction_count"],
                         max(0.8, pat["confidence"]),
                         pat["id"], time.time(), time.time()),
                    )
                    promoted += 1
                except sqlite3.IntegrityError:
                    # Duplicate principle text
                    pass

        return promoted

    def record_contradiction(self, pattern_id: int):
        """Record that a pattern was found to be wrong in a case.

        Reduces confidence but never deletes — we learn from contradictions too.
        """
        self.conn.execute(
            """UPDATE tier2_patterns
               SET contradiction_count = contradiction_count + 1,
                   confidence = CAST(support_count AS REAL) /
                                (support_count + contradiction_count + 1)
               WHERE id = ?""",
            (pattern_id,),
        )
        self.conn.commit()

    # ════════════════════════════════════════════════════════════════
    # KEYWORD EXTRACTION (for triggers and similarity search)
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract medical-relevant keywords from text.

        Unicode-aware: supports Turkish (ş, ğ, ü, ö, ç, ı, İ) and other scripts.
        Tokenize, remove stopwords, keep terms >= 3 chars.
        Returns deduplicated list sorted by frequency (most common first).
        """
        if not text:
            return []

        # Normalize and tokenize — \w matches Unicode word chars including Turkish
        text_lower = text.lower()
        tokens = re.findall(r'[\w]+(?:[-/][\w]+)*', text_lower, re.UNICODE)

        # Filter: remove stopwords, keep >= 3 chars
        filtered = [
            t for t in tokens
            if t not in _MEDICAL_STOPWORDS and len(t) >= 3
        ]

        # Return by frequency (most common first, deduplicated)
        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(30)]

    # ════════════════════════════════════════════════════════════════
    # STATS & UTILITIES
    # ════════════════════════════════════════════════════════════════

    def _tier1_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) as cnt FROM tier1_cases").fetchone()["cnt"]

    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        t1 = self._tier1_count()
        t2 = self.conn.execute("SELECT COUNT(*) as cnt FROM tier2_patterns").fetchone()["cnt"]
        t3 = self.conn.execute("SELECT COUNT(*) as cnt FROM tier3_principles").fetchone()["cnt"]
        consolidations = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM consolidation_log"
        ).fetchone()["cnt"]

        return {
            "tier1_cases": t1,
            "tier1_capacity": TIER1_CAPACITY,
            "tier1_fullness": f"{t1 / TIER1_CAPACITY:.0%}",
            "tier2_patterns": t2,
            "tier3_principles": t3,
            "consolidations_run": consolidations,
            "db_path": str(self.db_path),
        }

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("[MEMORY] Database connection closed.")
