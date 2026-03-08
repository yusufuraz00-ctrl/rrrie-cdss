"""Tavily Web Search API tool — agentic web search for medical information."""

from __future__ import annotations

from typing import Dict, List, Optional

import httpx

from config.api_config import TavilyConfig
from config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.rate_limiter import web_limiter, tool_cache

logger = get_logger(__name__)

_config = TavilyConfig()


async def web_search(
    query: str,
    search_depth: str = "advanced",
    include_domains: Optional[List[str]] = None,
) -> Dict:
    """Search the web for medical information using Tavily API.

    Args:
        query: Search query.
        search_depth: "basic" or "advanced" (deeper extraction).
        include_domains: Restrict to specific domains (e.g. ["who.int", "cdc.gov"]).

    Returns:
        Dict with search results including answer + individual results.
    """
    settings = get_settings()

    if not settings.tavily_api_key:
        logger.warning("tavily_no_api_key")
        return {"results": [], "answer": "", "error": "Tavily API key not configured"}

    payload: Dict = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": search_depth or _config.default_search_depth,
        "include_answer": True,
        "max_results": _config.default_max_results,
    }

    if include_domains:
        payload["include_domains"] = include_domains
    else:
        # Default: restrict to trusted medical domains
        payload["include_domains"] = list(_config.trusted_medical_domains)

    logger.info("web_search", query=query, depth=search_depth)

    # TTL cache check
    cache_key = tool_cache.make_key("tavily", query=query, depth=search_depth)
    cached = tool_cache.get(cache_key)
    if cached is not None:
        logger.info("web_search_cache_hit", query=query)
        return cached

    async with httpx.AsyncClient(timeout=_config.timeout) as client:
        try:
            async with web_limiter:
                resp = await client.post(_config.base_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("web_search_error", error=str(exc))
            return {"results": [], "answer": "", "error": str(exc)}

    results = []
    for r in data.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", "")[:500],  # Truncate for prompt brevity
            "score": r.get("score", 0),
        })

    answer = data.get("answer", "")

    logger.info("web_search_results", count=len(results), has_answer=bool(answer))

    result = {
        "query": query,
        "answer": answer,
        "results": results,
    }
    tool_cache.set(cache_key, result)
    return result
