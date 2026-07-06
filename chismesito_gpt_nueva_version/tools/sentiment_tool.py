# -*- coding: utf-8 -*-
"""tools/sentiment_tool.py — Análisis de sentimiento con PySentimiento."""

import logging
from langchain.tools import tool

logger = logging.getLogger(__name__)

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from pysentimiento import create_analyzer
        _analyzer = create_analyzer(task="sentiment", lang="es")
    return _analyzer


def analyze_sentiment_text(text: str) -> str:
    """Analiza sentimiento de un texto. Retorna: Positivo, Negativo, Neutral, Error."""
    try:
        analyzer = _get_analyzer()
        result = analyzer.predict(str(text))
        sentiment_map = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutral"}
        return sentiment_map.get(result.output, "Neutral")
    except Exception as e:
        logger.error(f"Sentiment error: {e}")
        return "Error"


@tool
def analyze_sentiment_batch(texts: list[str]) -> dict:
    """
    Analiza el sentimiento de una lista de textos.

    Args:
        texts: Lista de textos de comentarios

    Returns:
        {"success": bool, "data": [{"text": str, "sentiment": str}], "count": int, "error": str}
    """
    try:
        results = [{"text": t, "sentiment": analyze_sentiment_text(t)} for t in texts]
        return {"success": True, "data": results, "count": len(results), "error": None}
    except Exception as e:
        return {"success": False, "data": [], "count": 0, "error": str(e)}
