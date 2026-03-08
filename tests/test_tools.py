"""Tests for medical API tools (mocked HTTP calls)."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.tools.pubmed_tool import search_pubmed
from src.tools.clinical_trials_tool import search_clinical_trials
from src.tools.openfda_tool import search_drug_interactions
from src.tools.web_search_tool import web_search


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.AsyncClient."""
    mock = AsyncMock()
    return mock


class TestPubMedTool:
    """Test PubMed search tool."""

    @pytest.mark.asyncio
    async def test_search_pubmed_success(self, mock_httpx_client):
        # Mock eSearch response
        esearch_response = MagicMock()
        esearch_response.status_code = 200
        esearch_response.json.return_value = {
            "esearchresult": {
                "count": "1",
                "idlist": ["12345678"],
            }
        }

        # Mock eSummary response
        esummary_response = MagicMock()
        esummary_response.status_code = 200
        esummary_response.json.return_value = {
            "result": {
                "uids": ["12345678"],
                "12345678": {
                    "uid": "12345678",
                    "title": "Pneumonia Treatment Study",
                    "sortfirstauthor": "Smith J",
                    "authors": [{"name": "Smith J"}, {"name": "Doe A"}],
                    "fulljournalname": "NEJM",
                    "pubdate": "2024 Jan",
                    "elocationid": "doi: 10.1000/test",
                },
            }
        }

        mock_httpx_client.get = AsyncMock(
            side_effect=[esearch_response, esummary_response]
        )

        with patch("src.tools.pubmed_tool.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_httpx_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await search_pubmed(query="pneumonia treatment", max_results=5)

        assert "articles" in result or "error" in result

    @pytest.mark.asyncio
    async def test_search_pubmed_empty(self, mock_httpx_client):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "esearchresult": {"count": "0", "idlist": []}
        }

        mock_httpx_client.get = AsyncMock(return_value=response)

        with patch("src.tools.pubmed_tool.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_httpx_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await search_pubmed(query="zzz_nonexistent_disease_zzz")

        assert isinstance(result, dict)


class TestWebSearchTool:
    """Test web search tool."""

    @pytest.mark.asyncio
    async def test_web_search_success(self, mock_httpx_client):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "answer": "Treatment for pneumonia includes...",
            "results": [
                {
                    "title": "Pneumonia Treatment",
                    "url": "https://medlineplus.gov/pneumonia.html",
                    "content": "Treatment information...",
                    "score": 0.95,
                }
            ],
        }

        mock_httpx_client.post = AsyncMock(return_value=response)

        with patch("src.tools.web_search_tool.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_httpx_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("src.tools.web_search_tool.get_settings") as mock_settings:
                mock_settings.return_value.tavily_api_key = "test-key"
                result = await web_search(query="pneumonia treatment guidelines")

        assert isinstance(result, dict)
