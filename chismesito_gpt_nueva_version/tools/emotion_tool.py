# -*- coding: utf-8 -*-
"""tools/emotion_tool.py — Análisis de emoción (y sentimiento derivado).

Estrategia por entorno (para eficiencia + compatibilidad con ZeroGPU):

  • En HuggingFace Spaces (config.IS_SPACES == True):
        Se carga el modelo DIRECTO del Hub con `transformers.pipeline(device=-1)`.
        Forzar CPU evita inicializar CUDA en el proceso principal, lo cual es
        obligatorio en ZeroGPU (evita el error "Low-level CUDA init reached").

  • En local / CPU (IS_SPACES == False):
        Se usa la librería `pysentimiento` (create_analyzer), que trae su propio
        preprocesado nativo de tweets en español → mejor calidad.

En AMBOS entornos se corre UN SOLO modelo (emoción) y el sentimiento se DERIVA
de la emoción dominante (evita la "carga doble" y cualquier costo de LLM).

Modelo: pysentimiento/robertuito-emotion-analysis (español, RoBERTuito).
"""

import logging
import re

from langchain.tools import tool
from config import IS_SPACES

logger = logging.getLogger(__name__)

# Etiquetas del modelo (inglés) → español (coinciden con el dashboard).
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

# Sentimiento derivado de la emoción dominante (sin un 2º modelo).
EMOTION_TO_SENTIMENT = {
    "Alegria":  "Positivo",
    "Tristeza": "Negativo",
    "Enojo":    "Negativo",
    "Miedo":    "Negativo",
    "Asco":     "Negativo",
    "Sorpresa": "Neutral",   # ambigua → neutral
    "Neutral":  "Neutral",
}

_backend = None   # callable: list[str] -> list[str] (etiquetas de emoción en inglés)


def _preprocess(text: str) -> str:
    """Preprocesado ligero (solo para la ruta transformers): menciones y URLs."""
    t = str(text) if text else " "
    t = re.sub(r"@\w+", "@usuario", t)
    t = re.sub(r"https?://\S+", "url", t)
    return t.strip() or " "


def _build_backend():
    """Crea el backend de emoción según el entorno (HF transformers vs local pysentimiento)."""
    if IS_SPACES:
        from transformers import pipeline
        logger.info("Emoción: cargando robertuito-emotion-analysis vía transformers (HF, CPU)...")
        pipe = pipeline(
            "text-classification",
            model="pysentimiento/robertuito-emotion-analysis",
            device=-1,        # CPU → no inicializa CUDA en el proceso principal (ZeroGPU-safe)
            batch_size=16,
        )

        def _run(texts: list[str]) -> list[str]:
            results = pipe([_preprocess(t) for t in texts], batch_size=16)
            labels = []
            for r in results:
                item = r[0] if isinstance(r, list) else r   # top-1
                labels.append(item["label"])
            return labels

        logger.info("Emoción (transformers) lista.")
        return _run

    # Local / CPU → librería pysentimiento (mejor preprocesado nativo)
    from pysentimiento import create_analyzer
    logger.info("Emoción: cargando pysentimiento emotion analyzer (local)...")
    analyzer = create_analyzer(task="emotion", lang="es")

    def _run(texts: list[str]) -> list[str]:
        preds = analyzer.predict([str(t) for t in texts])
        if not isinstance(preds, list):
            preds = [preds]
        return [p.output for p in preds]

    logger.info("Emoción (pysentimiento) lista.")
    return _run


def _get_backend():
    global _backend
    if _backend is None:
        _backend = _build_backend()
    return _backend


def analyze_emotions_batch(texts: list[str]) -> list[tuple[str, str]]:
    """Clasifica una lista de textos con UNA sola pasada del modelo de emoción.

    Devuelve una lista de tuplas (emocion_es, sentimiento_es), donde el
    sentimiento se deriva de la emoción dominante. Robusto: ante un error
    devuelve ('Error', 'Error') por cada texto.
    """
    if not texts:
        return []
    try:
        labels = _get_backend()(list(texts))
        out = []
        for lab in labels:
            emo = EMOTION_MAP.get(lab, "Neutral")
            out.append((emo, EMOTION_TO_SENTIMENT.get(emo, "Neutral")))
        return out
    except Exception as e:
        logger.error(f"Emotion batch error: {e}")
        return [("Error", "Error")] * len(texts)


def analyze_emotion_text(text: str) -> str:
    """Emoción de un texto individual (delega en el batch)."""
    res = analyze_emotions_batch([text])
    return res[0][0] if res else "Error"


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
        pairs = analyze_emotions_batch(texts)
        results = [{"text": t, "emotion": e} for t, (e, _s) in zip(texts, pairs)]
        return {"success": True, "data": results, "count": len(results), "error": None}
    except Exception as e:
        return {"success": False, "data": [], "count": 0, "error": str(e)}
