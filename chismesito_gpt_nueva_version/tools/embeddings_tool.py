# -*- coding: utf-8 -*-
"""tools/embeddings_tool.py — Embeddings Gemini para pgvector."""

import logging
from langchain.tools import tool
from config import GEMINI_API_KEY
from llm_manager import discover_embedding_model

logger = logging.getLogger(__name__)

_client = None
_embedding_model: str | None = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _get_model() -> str | None:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = discover_embedding_model()
    return _embedding_model


def embed_single(text: str) -> list[float]:
    """Genera embedding para un texto (uso interno e importado por analyzer.py)."""
    try:
        client = _get_client()
        model = _get_model()
        if not model:
            return []

        result = client.models.embed_content(model=model, contents=[text])
        if hasattr(result, "embeddings") and result.embeddings:
            # Registrar costo del embedding (tokens de entrada estimados).
            try:
                from utils.cost_tracker import record_embedding, estimate_tokens
                record_embedding(estimate_tokens(text), model)
            except Exception as e:
                logger.debug(f"No se pudo registrar costo de embedding: {e}")
            return list(result.embeddings[0].values)
        return []
    except Exception as e:
        logger.error(f"Embed error: {e}")
        return []


@tool
def embed_comments(texts: list[str]) -> dict:
    """Genera embeddings para una lista de comentarios."""
    if not GEMINI_API_KEY:
        return {"success": False, "data": [], "count": 0, "error": "GEMINI_API_KEY no configurada"}

    try:
        results = []
        for i, text in enumerate(texts):
            emb = embed_single(text)
            results.append({"index": i, "embedding": emb})
        return {"success": True, "data": results, "count": len(results), "error": None}
    except Exception as e:
        return {"success": False, "data": [], "count": 0, "error": str(e)}
