"""Medical Knowledge Ingestor — Textbook/Guideline → Vector Database.

Automatically chunks medical textbooks and clinical guidelines, then stores
them in a SQLite-backed vector-like search system using FTS5 for retrieval.

NO external vector database dependency (ChromaDB/FAISS not required).
Uses SQLite FTS5 which is already in the project — zero new dependencies.

Usage:
  ingestor = KnowledgeIngestor()
  ingestor.ingest("data/textbooks/harrisons_nephrology.md")
  # → 847 chunks indexed and searchable

  results = ingestor.search("NSAID nephrotoxicity CKD", max_results=5)
  # → Returns relevant textbook passages

Supported formats:
  - Markdown (.md)
  - Plain text (.txt)
  - PDF → planned for future (requires PyMuPDF)

Architecture inspired by LLM-AMT (LLMs Augmented with Medical Textbooks, 2026).
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger("rrrie-cdss.knowledge")

# ── Configuration ───────────────────────────────────────────────────
CHUNK_SIZE = 512            # Target tokens per chunk (~4 chars/token)
CHUNK_OVERLAP = 64          # Token overlap between chunks for context continuity
MAX_CHUNKS_PER_FILE = 2000  # Safety cap per file
KB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge"
KB_DB_PATH = KB_DIR / "knowledge.db"
TEXTBOOK_DIR = KB_DIR / "textbooks"


@dataclass
class KBSearchResult:
    """A search result from the knowledge base."""
    chunk_id: int
    source_file: str
    section: str
    content: str
    rank: float             # FTS5 BM25 rank (lower = more relevant)


class KnowledgeIngestor:
    """Medical Knowledge Base — Textbook ingestion and FTS5-based retrieval.
    
    Uses SQLite FTS5 for full-text search. Zero external dependencies.
    """

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else KB_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        TEXTBOOK_DIR.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        logger.info("[KB] Knowledge Base initialized. DB: %s", self.db_path)

    def _init_schema(self):
        """Create KB tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS kb_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                section TEXT DEFAULT '',
                content TEXT NOT NULL,
                chunk_index INTEGER DEFAULT 0,
                created_at REAL DEFAULT 0.0
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts USING fts5(
                content,
                section,
                source_file,
                content='kb_chunks',
                content_rowid='id'
            );

            CREATE TRIGGER IF NOT EXISTS kb_ai AFTER INSERT ON kb_chunks BEGIN
                INSERT INTO kb_fts(rowid, content, section, source_file)
                VALUES (new.id, new.content, new.section, new.source_file);
            END;

            CREATE TRIGGER IF NOT EXISTS kb_ad AFTER DELETE ON kb_chunks BEGIN
                INSERT INTO kb_fts(kb_fts, rowid, content, section, source_file)
                VALUES ('delete', old.id, old.content, old.section, old.source_file);
            END;

            -- Track which files have been ingested
            CREATE TABLE IF NOT EXISTS kb_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                file_hash TEXT NOT NULL,
                chunks_count INTEGER DEFAULT 0,
                ingested_at REAL DEFAULT 0.0
            );
        """)
        self.conn.commit()

    def ingest(self, file_path: str | Path) -> dict:
        """Ingest a textbook/guideline file into the knowledge base.
        
        Supports: .md, .txt files
        
        Returns:
            dict with ingestion stats
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix.lower() not in (".md", ".txt"):
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .md or .txt")

        # Check if already ingested (same hash)
        content = path.read_text(encoding="utf-8", errors="replace")
        file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        existing = self.conn.execute(
            "SELECT file_hash FROM kb_sources WHERE filename = ?",
            (path.name,),
        ).fetchone()
        
        if existing and existing["file_hash"] == file_hash:
            logger.info("[KB] File '%s' already ingested (hash match), skipping.", path.name)
            return {"status": "skipped", "reason": "already_ingested", "chunks": 0}

        # If file changed, re-ingest (delete old chunks first)
        if existing:
            self._delete_source(path.name)
            logger.info("[KB] File '%s' changed, re-ingesting.", path.name)

        # Chunk the content
        chunks = self._chunk_content(content, path.name)
        
        if not chunks:
            return {"status": "empty", "chunks": 0}

        # Store chunks
        for i, chunk in enumerate(chunks):
            self.conn.execute(
                """INSERT INTO kb_chunks
                   (source_file, source_hash, section, content, chunk_index, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (path.name, file_hash, chunk["section"], chunk["content"], i, time.time()),
            )
        
        # Record source
        self.conn.execute(
            """INSERT OR REPLACE INTO kb_sources
               (filename, file_hash, chunks_count, ingested_at)
               VALUES (?, ?, ?, ?)""",
            (path.name, file_hash, len(chunks), time.time()),
        )
        self.conn.commit()

        logger.info("[KB] Ingested '%s': %d chunks", path.name, len(chunks))
        return {"status": "ingested", "file": path.name, "chunks": len(chunks)}

    def ingest_directory(self, dir_path: str | Path | None = None) -> dict:
        """Ingest all .md and .txt files from a directory.
        
        Defaults to data/knowledge/textbooks/ if no path specified.
        """
        directory = Path(dir_path) if dir_path else TEXTBOOK_DIR
        if not directory.exists():
            return {"status": "no_directory", "files_processed": 0}

        results = []
        for ext in ("*.md", "*.txt"):
            for f in sorted(directory.glob(ext)):
                try:
                    result = self.ingest(f)
                    results.append({"file": f.name, **result})
                except Exception as exc:
                    logger.warning("[KB] Failed to ingest '%s': %s", f.name, exc)
                    results.append({"file": f.name, "status": "error", "error": str(exc)})

        total = sum(1 for r in results if r["status"] == "ingested")
        total_chunks = sum(r.get("chunks", 0) for r in results)
        logger.info("[KB] Directory ingestion: %d files, %d chunks", total, total_chunks)
        return {"files_processed": len(results), "files_ingested": total, "total_chunks": total_chunks}

    def search(self, query: str, max_results: int = 5) -> list[KBSearchResult]:
        """Search the knowledge base using FTS5 full-text search.
        
        Args:
            query: Medical search query (e.g., "NSAID nephrotoxicity CKD")
            max_results: Maximum results to return
            
        Returns:
            List of KBSearchResult sorted by relevance
        """
        # Tokenize and clean query for FTS5
        words = re.findall(r'[a-zA-Z0-9]{3,}', query.lower())
        if not words:
            return []

        # Build FTS5 query with OR
        safe_terms = [f'"{w}"' for w in words[:10]]
        fts_query = " OR ".join(safe_terms)

        try:
            rows = self.conn.execute(
                """SELECT c.*, rank FROM kb_fts f
                   JOIN kb_chunks c ON c.id = f.rowid
                   WHERE kb_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, max_results),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("[KB] FTS5 search failed: %s", exc)
            return []

        return [
            KBSearchResult(
                chunk_id=row["id"],
                source_file=row["source_file"],
                section=row["section"],
                content=row["content"],
                rank=row["rank"],
            )
            for row in rows
        ]

    def format_for_prompt(self, results: list[KBSearchResult], max_tokens: int = 500) -> str:
        """Format search results for LLM prompt injection.
        
        Targets ~500 tokens to avoid prompt bloat.
        """
        if not results:
            return ""

        parts = ["📖 MEDICAL KNOWLEDGE BASE (textbook references):\n"]
        total_chars = 0
        max_chars = max_tokens * 4  # ~4 chars per token

        for r in results:
            chunk = f"[{r.source_file} — {r.section}]\n{r.content}\n---\n"
            if total_chars + len(chunk) > max_chars:
                break
            parts.append(chunk)
            total_chars += len(chunk)

        return "\n".join(parts)

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        chunks = self.conn.execute("SELECT COUNT(*) as cnt FROM kb_chunks").fetchone()["cnt"]
        sources = self.conn.execute("SELECT COUNT(*) as cnt FROM kb_sources").fetchone()["cnt"]
        return {"total_chunks": chunks, "total_sources": sources}

    def _chunk_content(self, content: str, filename: str) -> list[dict]:
        """Split content into overlapping chunks with section tracking."""
        chunks = []
        current_section = filename  # Default section = filename

        # Split by markdown headers for section tracking
        lines = content.split("\n")
        buffer = []
        buffer_section = current_section

        for line in lines:
            # Detect markdown headers
            header_match = re.match(r'^(#{1,3})\s+(.+)', line)
            if header_match:
                # Flush buffer if it has content
                if buffer:
                    text = "\n".join(buffer).strip()
                    if len(text) > 20:  # Skip tiny fragments
                        chunks.extend(self._split_text_to_chunks(text, buffer_section))
                    buffer = []
                buffer_section = header_match.group(2).strip()[:100]
            
            buffer.append(line)

        # Flush remaining buffer
        if buffer:
            text = "\n".join(buffer).strip()
            if len(text) > 20:
                chunks.extend(self._split_text_to_chunks(text, buffer_section))

        # Apply safety cap
        if len(chunks) > MAX_CHUNKS_PER_FILE:
            logger.warning("[KB] File '%s' produced %d chunks, capping at %d",
                         filename, len(chunks), MAX_CHUNKS_PER_FILE)
            chunks = chunks[:MAX_CHUNKS_PER_FILE]

        return chunks

    @staticmethod
    def _split_text_to_chunks(
        text: str, section: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
    ) -> list[dict]:
        """Split text into overlapping chunks of ~chunk_size tokens."""
        char_size = chunk_size * 4  # ~4 chars per token
        char_overlap = overlap * 4

        chunks = []
        start = 0
        while start < len(text):
            end = start + char_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({"section": section, "content": chunk_text})
            start += char_size - char_overlap

        return chunks

    def _delete_source(self, filename: str):
        """Delete all chunks from a source file."""
        self.conn.execute("DELETE FROM kb_chunks WHERE source_file = ?", (filename,))
        self.conn.execute("DELETE FROM kb_sources WHERE filename = ?", (filename,))
        self.conn.commit()
