# -*- coding: utf-8 -*-
"""tools/search_tool.py — Búsqueda web con SerpAPI."""

import logging
from langchain.tools import tool
from config import SERPAPI_API_KEY

logger = logging.getLogger(__name__)

try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None


@tool
def search_web(query: str, num_results: int = 5) -> dict:
    """
    Busca en la web usando SerpAPI para encontrar links reales de redes sociales.

    Usa esta tool SIEMPRE antes de extraer comentarios. No inventes URLs.

    Args:
        query: Termino de busqueda, incluye site:reddit.com o site:youtube.com
        num_results: Numero maximo de resultados

    Returns:
        {"success": bool, "data": [{"title": str, "link": str, "snippet": str}], "count": int, "error": str}
    """
    if not SERPAPI_API_KEY:
        return {"success": False, "data": [], "count": 0, "error": "SERPAPI_API_KEY no configurada"}
    if GoogleSearch is None:
        return {"success": False, "data": [], "count": 0, "error": "serpapi no instalado (pip install google-search-results)"}

    try:
        params = {
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": num_results,
            "engine": "google",
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        organic = results.get("organic_results", [])
        data = [
            {
                "title": r.get("title", ""),
                "link": r.get("link", ""),
                "snippet": r.get("snippet", ""),
            }
            for r in organic[:num_results]
        ]
        logger.info(f"SerpAPI: {len(data)} resultados para '{query}'")
        return {"success": True, "data": data, "count": len(data), "error": None}

    except Exception as e:
        logger.error(f"SerpAPI error: {e}")
        return {"success": False, "data": [], "count": 0, "error": str(e)}
