"""Semantic Scholar API tool — citation-aware academic paper search.

Semantic Scholar indexes 200M+ papers with citation graphs, influence scores,
and abstracts. Useful for finding highly-cited evidence and systematic reviews.
Free tier: 100 requests/5 minutes (no key), 1 request/second.

API docs: https://api.semanticscholar.org/api-docs/
"""

from __future__ import annotations

from typing import Dict, List, Optional

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_TIMEOUT = 30.0
_HEADERS = {
    "User-Agent": "RRRIE-CDSS/1.0",
}

# Fields to request from the API
_PAPER_FIELDS = (
    "paperId,title,abstract,authors,year,citationCount,"
    "influentialCitationCount,journal,externalIds,url,publicationTypes"
)


async def search_semantic_scholar(
    query: str,
    max_results: int = 5,
    year_range: Optional[str] = None,
    publication_types: Optional[List[str]] = None,
) -> Dict:
    """Search Semantic Scholar for academic papers.

    Args:
        query: Search query (natural language or structured).
        max_results: Maximum papers to return (default 5, max 100).
        year_range: Year range filter, e.g. "2020-2025" or "2023-" (from 2023 onward).
        publication_types: Filter by type: "Review", "ClinicalTrial", "MetaAnalysis",
                          "CaseReport", "JournalArticle".

    Returns:
        Dict with "papers" list, "total_found" count, and optional "error".
    """
    params: Dict = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": _PAPER_FIELDS,
    }

    if year_range:
        params["year"] = year_range

    if publication_types:
        params["publicationTypes"] = ",".join(publication_types)

    logger.info("semantic_scholar_search", query=query)

    async with httpx.AsyncClient(
        timeout=_TIMEOUT,
        headers=_HEADERS,
    ) as client:
        try:
            resp = await client.get(f"{_BASE_URL}/paper/search", params=params)
            # Retry once on 429 (rate limit) after a short wait
            if resp.status_code == 429:
                import asyncio as _asyncio
                logger.warning("semantic_scholar_rate_limit — retrying in 3s")
                await _asyncio.sleep(3)
                resp = await client.get(f"{_BASE_URL}/paper/search", params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("semantic_scholar_error", error=str(exc))
            return {"papers": [], "total_found": 0, "error": str(exc)}

    total_found = data.get("total", 0)
    raw_papers = data.get("data", [])

    papers: List[Dict] = []
    for p in raw_papers:
        # Extract external IDs
        ext_ids = p.get("externalIds", {}) or {}
        pmid = ext_ids.get("PubMed", "")
        doi = ext_ids.get("DOI", "")

        # Extract authors
        authors = []
        for a in (p.get("authors") or [])[:5]:
            name = a.get("name", "")
            if name:
                authors.append(name)

        papers.append({
            "paper_id": p.get("paperId", ""),
            "title": p.get("title", ""),
            "abstract": p.get("abstract", "") or "",
            "authors": authors,
            "year": p.get("year"),
            "journal": (p.get("journal") or {}).get("name", ""),
            "citation_count": p.get("citationCount", 0),
            "influential_citations": p.get("influentialCitationCount", 0),
            "publication_types": p.get("publicationTypes") or [],
            "pmid": pmid,
            "doi": doi,
            "url": p.get("url", ""),
        })

    # Sort by citation count (most cited first) for clinical relevance
    papers.sort(key=lambda x: x.get("citation_count", 0), reverse=True)

    logger.info(
        "semantic_scholar_results",
        count=len(papers),
        total=total_found,
        abstracts=sum(1 for p in papers if p.get("abstract")),
    )
    return {"papers": papers, "total_found": total_found}
