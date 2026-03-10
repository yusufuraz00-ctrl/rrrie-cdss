"""Evidence Memory Store — Persistent R2 Research Knowledge Base.

Every R2 research result (PubMed articles, Wikipedia summaries, web search
answers) is stored here. Future pipeline runs retrieve relevant past research
as ADDITIVE context — supplementing, not replacing, live R2 searches.

Also stores diagnostic patterns learned from successful cases:
  symptom_set → diagnosis mapping, accumulated over time.

Uses SQLite FTS5 — zero external dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("rrrie-cdss.evidence")

# ── Configuration ─────────────────────────────────────────────────
EVIDENCE_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge" / "evidence.db"


@dataclass
class EvidenceResult:
    """A single evidence item retrieved from memory."""
    id: int
    source: str          # "pubmed", "europe_pmc", "wikipedia", "web", etc.
    title: str
    abstract: str
    diagnosis_context: str  # Which diagnosis this was found for
    pmid: str
    case_count: int       # How many cases have used this evidence
    rank: float           # FTS5 relevance rank


@dataclass
class DiagnosticPattern:
    """A learned symptom→diagnosis mapping."""
    id: int
    diagnosis: str
    symptoms: list[str]
    confidence: float
    case_count: int


class EvidenceStore:
    """Persistent Evidence Memory — stores and retrieves R2 research findings.

    Features:
      - Deduplicates by content_hash (same article/source never stored twice)
      - Tracks case_count (how many cases used each evidence item)
      - FTS5 full-text search for retrieval
      - Diagnostic pattern storage for adaptive L3
    """

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else EVIDENCE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        """Create evidence tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS evidence_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                query TEXT DEFAULT '',
                diagnosis_context TEXT DEFAULT '',
                title TEXT DEFAULT '',
                pmid TEXT DEFAULT '',
                url TEXT DEFAULT '',
                abstract TEXT DEFAULT '',
                content_hash TEXT UNIQUE NOT NULL,
                case_count INTEGER DEFAULT 1,
                created_at REAL DEFAULT 0.0,
                last_used_at REAL DEFAULT 0.0
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
                title, abstract, diagnosis_context, query,
                content='evidence_items',
                content_rowid='id'
            );

            CREATE TRIGGER IF NOT EXISTS ev_ai AFTER INSERT ON evidence_items BEGIN
                INSERT INTO evidence_fts(rowid, title, abstract, diagnosis_context, query)
                VALUES (new.id, new.title, new.abstract, new.diagnosis_context, new.query);
            END;

            CREATE TRIGGER IF NOT EXISTS ev_ad AFTER DELETE ON evidence_items BEGIN
                INSERT INTO evidence_fts(evidence_fts, rowid, title, abstract, diagnosis_context, query)
                VALUES ('delete', old.id, old.title, old.abstract, old.diagnosis_context, old.query);
            END;

            -- Diagnostic patterns learned from successful cases
            CREATE TABLE IF NOT EXISTS diagnostic_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                diagnosis TEXT NOT NULL,
                symptoms_json TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                case_count INTEGER DEFAULT 1,
                created_at REAL DEFAULT 0.0,
                updated_at REAL DEFAULT 0.0,
                UNIQUE(diagnosis, symptoms_json)
            );

            -- Diagnosis frequency tracking for experience-weighted ranking
            CREATE TABLE IF NOT EXISTS diagnosis_frequency (
                diagnosis TEXT PRIMARY KEY,
                total_cases INTEGER DEFAULT 1,
                avg_confidence REAL DEFAULT 0.0,
                last_seen_at REAL DEFAULT 0.0
            );
        """)
        self.conn.commit()
        logger.info("[EVIDENCE] Evidence Store initialized. DB: %s", self.db_path)

    # ══════════════════════════════════════════════════════════════
    # STORAGE — Called after R2 completes
    # ══════════════════════════════════════════════════════════════

    def store_r2_results(
        self,
        r2_evidence: list[dict],
        diagnosis: str,
    ) -> dict:
        """Store R2 evidence items into persistent memory.

        Deduplicates by content hash. Increments case_count for existing items.

        Returns:
            dict with storage stats.
        """
        stored = 0
        updated = 0
        skipped = 0
        now = time.time()

        for ev in r2_evidence:
            source = ev.get("source", "unknown")
            query = ev.get("query", "")

            # Extract articles (PubMed, Europe PMC)
            for article in ev.get("articles", []):
                title = article.get("title", "")
                abstract = article.get("abstract", "")[:2000]
                pmid = str(article.get("pmid", ""))
                if not title and not abstract:
                    continue

                content_hash = self._hash(f"{source}:{pmid}:{title}")
                result = self._upsert_evidence(
                    source=source, query=query, diagnosis_context=diagnosis,
                    title=title, pmid=pmid, url="", abstract=abstract,
                    content_hash=content_hash, now=now,
                )
                if result == "inserted":
                    stored += 1
                elif result == "updated":
                    updated += 1
                else:
                    skipped += 1

            # Extract web search answers
            answer = ev.get("answer", "")
            if answer:
                content_hash = self._hash(f"web:{query}:{answer[:200]}")
                result = self._upsert_evidence(
                    source="web_search", query=query, diagnosis_context=diagnosis,
                    title=f"Web: {query[:100]}", pmid="", url="",
                    abstract=answer[:2000], content_hash=content_hash, now=now,
                )
                if result == "inserted":
                    stored += 1
                elif result == "updated":
                    updated += 1

            # Extract Wikipedia summaries
            summary = ev.get("summary", "")
            if summary:
                content_hash = self._hash(f"wiki:{query}:{summary[:200]}")
                result = self._upsert_evidence(
                    source="wikipedia", query=query, diagnosis_context=diagnosis,
                    title=f"Wikipedia: {ev.get('page_title', query[:80])}",
                    pmid="", url="", abstract=summary[:2000],
                    content_hash=content_hash, now=now,
                )
                if result == "inserted":
                    stored += 1
                elif result == "updated":
                    updated += 1

            # Extract Semantic Scholar papers
            for paper in ev.get("papers", []):
                title = paper.get("title", "")
                if not title:
                    continue
                content_hash = self._hash(f"ss:{title}")
                result = self._upsert_evidence(
                    source="semantic_scholar", query=query,
                    diagnosis_context=diagnosis, title=title, pmid="",
                    url=paper.get("url", ""), abstract=paper.get("abstract", "")[:2000],
                    content_hash=content_hash, now=now,
                )
                if result == "inserted":
                    stored += 1
                elif result == "updated":
                    updated += 1

        self.conn.commit()
        logger.info(
            "[EVIDENCE] Stored R2 results: %d new, %d updated, %d skipped (diagnosis: %s)",
            stored, updated, skipped, diagnosis,
        )
        return {"stored": stored, "updated": updated, "skipped": skipped}

    def _upsert_evidence(
        self, *, source: str, query: str, diagnosis_context: str,
        title: str, pmid: str, url: str, abstract: str,
        content_hash: str, now: float,
    ) -> str:
        """Insert new evidence or increment case_count if exists."""
        existing = self.conn.execute(
            "SELECT id FROM evidence_items WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()

        if existing:
            self.conn.execute(
                "UPDATE evidence_items SET case_count = case_count + 1, last_used_at = ? WHERE id = ?",
                (now, existing["id"]),
            )
            return "updated"

        try:
            self.conn.execute(
                """INSERT INTO evidence_items
                   (source, query, diagnosis_context, title, pmid, url, abstract, content_hash, created_at, last_used_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (source, query, diagnosis_context, title, pmid, url, abstract, content_hash, now, now),
            )
            return "inserted"
        except sqlite3.IntegrityError:
            return "skipped"

    # ══════════════════════════════════════════════════════════════
    # RETRIEVAL — Called before R3 for additive context
    # ══════════════════════════════════════════════════════════════

    def search_past_evidence(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[EvidenceResult]:
        """Search past research findings using FTS5."""
        words = re.findall(r'[a-zA-Z0-9]{3,}', query.lower())
        if not words:
            return []

        safe_terms = [f'"{w}"' for w in words[:10]]
        fts_query = " OR ".join(safe_terms)

        try:
            rows = self.conn.execute(
                """SELECT e.*, rank FROM evidence_fts f
                   JOIN evidence_items e ON e.id = f.rowid
                   WHERE evidence_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, max_results),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("[EVIDENCE] FTS5 search failed: %s", exc)
            return []

        return [
            EvidenceResult(
                id=row["id"], source=row["source"], title=row["title"],
                abstract=row["abstract"], diagnosis_context=row["diagnosis_context"],
                pmid=row["pmid"], case_count=row["case_count"], rank=row["rank"],
            )
            for row in rows
        ]

    def format_for_prompt(
        self, results: list[EvidenceResult], max_chars: int = 1500,
    ) -> str:
        """Format past evidence for R3 prompt injection."""
        if not results:
            return ""

        parts = ["📚 PAST RESEARCH MEMORY (from similar previous cases):\n"]
        total = 0

        for r in results:
            abstract_preview = r.abstract[:300] + "..." if len(r.abstract) > 300 else r.abstract
            block = (
                f"[{r.source}] {r.title}\n"
                f"  Context: {r.diagnosis_context} | Used in {r.case_count} case(s)\n"
            )
            if r.pmid:
                block += f"  PMID: {r.pmid}\n"
            if abstract_preview:
                block += f"  Summary: {abstract_preview}\n"
            block += "---\n"

            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)

        return "\n".join(parts)

    # ══════════════════════════════════════════════════════════════
    # DIAGNOSTIC PATTERNS — Learned from successful cases
    # ══════════════════════════════════════════════════════════════

    def store_diagnostic_pattern(
        self,
        diagnosis: str,
        symptoms: list[str],
        confidence: float,
    ) -> None:
        """Store or reinforce a symptom→diagnosis pattern.

        Called after every successful FINALIZE.
        """
        now = time.time()
        symptoms_sorted = sorted(set(s.lower().strip() for s in symptoms if s.strip()))
        if not symptoms_sorted or not diagnosis:
            return

        symptoms_json = json.dumps(symptoms_sorted, ensure_ascii=False)

        existing = self.conn.execute(
            "SELECT id, case_count, confidence FROM diagnostic_patterns WHERE diagnosis = ? AND symptoms_json = ?",
            (diagnosis, symptoms_json),
        ).fetchone()

        if existing:
            # Reinforce: increment count, update running average confidence
            new_count = existing["case_count"] + 1
            new_conf = (existing["confidence"] * existing["case_count"] + confidence) / new_count
            self.conn.execute(
                "UPDATE diagnostic_patterns SET case_count = ?, confidence = ?, updated_at = ? WHERE id = ?",
                (new_count, new_conf, now, existing["id"]),
            )
        else:
            self.conn.execute(
                """INSERT INTO diagnostic_patterns (diagnosis, symptoms_json, confidence, case_count, created_at, updated_at)
                   VALUES (?, ?, ?, 1, ?, ?)""",
                (diagnosis, symptoms_json, confidence, now, now),
            )

        # Update diagnosis frequency
        self.conn.execute(
            """INSERT INTO diagnosis_frequency (diagnosis, total_cases, avg_confidence, last_seen_at)
               VALUES (?, 1, ?, ?)
               ON CONFLICT(diagnosis) DO UPDATE SET
                 total_cases = total_cases + 1,
                 avg_confidence = (avg_confidence * total_cases + ?) / (total_cases + 1),
                 last_seen_at = ?""",
            (diagnosis, confidence, now, confidence, now),
        )

        self.conn.commit()
        logger.info(
            "[EVIDENCE] Diagnostic pattern stored: %s (%d symptoms, conf=%.2f)",
            diagnosis, len(symptoms_sorted), confidence,
        )

    def get_relevant_patterns(
        self,
        symptoms: list[str],
        max_patterns: int = 5,
    ) -> list[DiagnosticPattern]:
        """Retrieve diagnostic patterns matching the current symptoms.

        Uses keyword overlap scoring — patterns that share more symptoms
        with the current case rank higher.
        """
        current_symptoms = set(s.lower().strip() for s in symptoms if s.strip())
        if not current_symptoms:
            return []

        all_patterns = self.conn.execute(
            "SELECT * FROM diagnostic_patterns ORDER BY case_count DESC LIMIT 100",
        ).fetchall()

        scored = []
        for row in all_patterns:
            pattern_symptoms = set(json.loads(row["symptoms_json"]))
            overlap = len(current_symptoms & pattern_symptoms)
            if overlap == 0:
                continue
            # Jaccard-like score weighted by case_count
            union = len(current_symptoms | pattern_symptoms)
            score = (overlap / union) * (1 + row["case_count"] * 0.1)
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            DiagnosticPattern(
                id=row["id"],
                diagnosis=row["diagnosis"],
                symptoms=json.loads(row["symptoms_json"]),
                confidence=row["confidence"],
                case_count=row["case_count"],
            )
            for _, row in scored[:max_patterns]
        ]

    def format_patterns_for_l3(self, patterns: list[DiagnosticPattern]) -> str:
        """Format dynamic patterns for L3 prompt injection."""
        if not patterns:
            return ""

        lines = ["\nLEARNED PATTERNS FROM PAST CASES (dynamic — system learned these):"]
        for p in patterns:
            symptom_str = " + ".join(p.symptoms[:6])
            lines.append(
                f"- {symptom_str} = {p.diagnosis} "
                f"(seen {p.case_count}x, avg conf: {p.confidence:.0%})"
            )
        return "\n".join(lines)

    def get_top_diagnoses(self, limit: int = 10) -> list[tuple[str, int, float]]:
        """Get most frequently diagnosed conditions for L5 Bayesian prior.

        Returns: list of (diagnosis, total_cases, avg_confidence)
        """
        rows = self.conn.execute(
            "SELECT diagnosis, total_cases, avg_confidence FROM diagnosis_frequency ORDER BY total_cases DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(row["diagnosis"], row["total_cases"], row["avg_confidence"]) for row in rows]

    def get_stats(self) -> dict:
        """Get evidence store statistics."""
        items = self.conn.execute("SELECT COUNT(*) as cnt FROM evidence_items").fetchone()["cnt"]
        patterns = self.conn.execute("SELECT COUNT(*) as cnt FROM diagnostic_patterns").fetchone()["cnt"]
        diagnoses = self.conn.execute("SELECT COUNT(*) as cnt FROM diagnosis_frequency").fetchone()["cnt"]
        return {"evidence_items": items, "diagnostic_patterns": patterns, "unique_diagnoses": diagnoses}

    @staticmethod
    def _hash(text: str) -> str:
        """Generate content hash for deduplication."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]
