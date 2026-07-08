# -*- coding: utf-8 -*-
"""tools/sentiment_tool.py — Sentimiento DERIVADO de la emoción.

Para evitar cargar un segundo modelo (y cualquier costo de LLM), el sentimiento
se infiere de la emoción dominante detectada por `emotion_tool` (un solo modelo
RoBERTuito). El mapeo emoción→sentimiento vive en `emotion_tool.EMOTION_TO_SENTIMENT`.

Nota: derivar el sentimiento de la emoción es una aproximación. Es preciso para
texto emotivo, pero opiniones "planas" (p.ej. quejas factuales sin carga
emocional) tienden a caer en Neutral. Si se necesita mayor precisión de
sentimiento, se puede cargar `pysentimiento/robertuito-sentiment-analysis`
directo con transformers (mismo patrón que emotion_tool, device=-1).
"""

import logging
from langchain.tools import tool

from tools.emotion_tool import analyze_emotions_batch

logger = logging.getLogger(__name__)


def analyze_sentiments_batch(texts: list[str]) -> list[str]:
    """Sentimiento (Positivo/Negativo/Neutral) por texto, derivado de la emoción."""
    return [s for (_e, s) in analyze_emotions_batch(texts)]


def analyze_sentiment_text(text: str) -> str:
    """Sentimiento de un texto individual. Retorna: Positivo, Negativo, Neutral, Error."""
    res = analyze_sentiments_batch([text])
    return res[0] if res else "Error"


@tool
def analyze_sentiment_batch(texts: list[str]) -> dict:
    """
    Analiza el sentimiento de una lista de textos (derivado de la emoción).

    Args:
        texts: Lista de textos de comentarios

    Returns:
        {"success": bool, "data": [{"text": str, "sentiment": str}], "count": int, "error": str}
    """
    try:
        sents = analyze_sentiments_batch(texts)
        results = [{"text": t, "sentiment": s} for t, s in zip(texts, sents)]
        return {"success": True, "data": results, "count": len(results), "error": None}
    except Exception as e:
        return {"success": False, "data": [], "count": 0, "error": str(e)}
