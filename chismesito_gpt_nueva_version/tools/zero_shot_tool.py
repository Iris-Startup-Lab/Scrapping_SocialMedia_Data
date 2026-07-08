# -*- coding: utf-8 -*-
"""
tools/zero_shot_tool.py — Clasificador zero-shot multilingüe.

Modelo: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
  - Soporte nativo para español (multilingual)
  - ~700 MB (vs 1.6 GB de BART)
  - ~1.0s/comentario en CPU · batch_size=8 reduce el tiempo total ~60%

Ejecución en CPU (device=-1) SIEMPRE:
  - Para los volúmenes de esta app (decenas de comentarios) la CPU es suficiente
    y evita el cupo (quota) limitado de HuggingFace ZeroGPU, que se agota rápido
    ("You have exceeded your ZeroGPU quota").
  - Correr en CPU (device=-1) tampoco inicializa CUDA en el proceso principal,
    lo cual es requisito en ZeroGPU.

Nota ZeroGPU: en hardware ZeroGPU, HF exige que exista ≥1 función @spaces.GPU al
arrancar. Definimos una SONDA mínima (nunca se invoca → no consume cupo) solo si
detectamos ZeroGPU, para que el arranque no falle. La clasificación real es CPU.
"""

import os
import logging
from langchain.tools import tool

logger = logging.getLogger(__name__)

# ¿Estamos en hardware ZeroGPU? HF define SPACES_ZERO_GPU en esos contenedores.
_IS_ZEROGPU = bool(os.getenv("SPACES_ZERO_GPU"))

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        logger.info("Cargando mDeBERTa zero-shot classifier (primera vez, puede tardar)...")
        _classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",  # multilingual, soporte español
            device=-1,      # CPU: suficiente para estos volúmenes y evita el cupo de ZeroGPU
            batch_size=8,   # procesa 8 comentarios a la vez → ~60% más rápido
        )
        logger.info("mDeBERTa zero-shot classifier cargado correctamente.")
    return _classifier


def classify_comments_batch(texts: list[str], candidate_labels: list[str]) -> list[str]:
    """Clasifica una lista de comentarios con UNA sola llamada (batch, CPU).
    Robusto: devuelve 'Error' por cada texto si algo falla."""
    if not texts:
        return []
    if not candidate_labels:
        return ["Sin categoria"] * len(texts)
    clean = [str(t) if t else " " for t in texts]
    try:
        classifier = _get_classifier()
        results = classifier(clean, candidate_labels, batch_size=8)
        if isinstance(results, dict):   # una sola entrada → dict en vez de lista
            results = [results]
        return [r["labels"][0] for r in results]
    except Exception as e:
        logger.error(f"Zero-shot batch error: {e}")
        return ["Error"] * len(texts)


def classify_comment(text: str, candidate_labels: list[str]) -> str:
    """Clasifica un comentario individual (delega en el batch)."""
    if not text or not candidate_labels:
        return "Sin categoria"
    res = classify_comments_batch([text], candidate_labels)
    return res[0] if res else "Error"


# ── Sonda ZeroGPU (solo en hardware ZeroGPU) ─────────────────────────────────
# HF exige ≥1 función @spaces.GPU detectada al arrancar. Esta sonda NUNCA se
# invoca (no consume cupo); solo existe para que el arranque no falle. Toda la
# inferencia real corre en CPU arriba.
if _IS_ZEROGPU:
    try:
        import spaces

        @spaces.GPU(duration=15)
        def _zerogpu_startup_probe():
            return True
    except Exception as e:
        logger.warning(f"No se pudo registrar la sonda ZeroGPU: {e}")


@tool
def zero_shot_classify(texts: list[str], candidate_labels: list[str]) -> dict:
    """
    Clasifica cada comentario en una categoria usando zero-shot classification.

    Args:
        texts: Lista de comentarios a clasificar
        candidate_labels: Lista de categorias candidatas (generadas por generate_categories)

    Returns:
        {"success": bool, "data": [{"text": str, "category": str}], "count": int, "error": str}
    """
    try:
        cats = classify_comments_batch(texts, candidate_labels)
        results = [{"text": t, "category": c} for t, c in zip(texts, cats)]
        return {"success": True, "data": results, "count": len(results), "error": None}
    except Exception as e:
        return {"success": False, "data": [], "count": 0, "error": str(e)}
