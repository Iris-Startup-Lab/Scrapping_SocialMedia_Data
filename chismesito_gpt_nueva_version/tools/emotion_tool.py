# -*- coding: utf-8 -*-
"""tools/emotion_tool.py — Análisis de emociones con PySentimiento."""

import logging
from langchain.tools import tool

logger = logging.getLogger(__name__)

_emotion_analyzer = None

EMOTION_MAP = {
    "joy": "Alegria",
    "sadness": "Tristeza",
    "anger": "Enojo",
    "fear": "Miedo",
    "surprise": "Sorpresa",
    "disgust": "Asco",
    "neutral": "Neutral",
    "others": "Neutral",
}


def _get_emotion_analyzer():
    global _emotion_analyzer
    if _emotion_analyzer is None:
        from pysentimiento import create_analyzer
        _emotion_analyzer = create_analyzer(task="emotion", lang="es")
    return _emotion_analyzer


def analyze_emotion_text(text: str) -> str:
    """Analiza emocion de un texto."""
    try:
        analyzer = _get_emotion_analyzer()
        result = analyzer.predict(str(text))
        return EMOTION_MAP.get(result.output, "Neutral")
    except Exception as e:
        logger.error(f"Emotion error: {e}")
        return "Error"


@tool
def analyze_emotion_batch(texts: list[str]) -> dict:
    """
    Analiza la emocion predominante de una lista de textos.

    Args:
        texts: Lista de textos de comentarios

    Returns:
        {"success": bool, "data": [{"text": str, "emotion": str}], "count": int, "error": str}
    """
    try:
        results = [{"text": t, "emotion": analyze_emotion_text(t)} for t in texts]
        return {"success": True, "data": results, "count": len(results), "error": None}
    except Exception as e:
        return {"success": False, "data": [], "count": 0, "error": str(e)}
