# -*- coding: utf-8 -*-
"""pipeline/rag.py — Busqueda semantica + chat RAG sobre comentarios."""

import logging
import pandas as pd
from db.vector import semantic_search
from tools.embeddings_tool import embed_single
from llm_manager import get_llm_response

logger = logging.getLogger(__name__)


def retrieve_context(query: str, session_id: str, user_id: str, top_k: int = 5) -> str:
    """Busca comentarios relevantes semanticamente en Supabase pgvector."""
    try:
        embedding = embed_single(query)
        if not embedding:
            return ""
        results = semantic_search(embedding, session_id, user_id, top_k)
        if not results:
            return ""
        return "\n---\n".join([r.get("comment", "") for r in results])
    except Exception as e:
        logger.error(f"RAG retrieve error: {e}")
        return ""


def rag_chat(
    question: str,
    session_id: str,
    user_id: str,
    chat_history: list[dict],
    model: str = "gemini-2.0-flash-lite",
    df_fallback: pd.DataFrame | None = None,
    stats: dict | None = None,
) -> str:
    """Responde usando RAG sobre Supabase, con fallback al DataFrame en memoria."""
    context = retrieve_context(question, session_id, user_id, top_k=10)

    # Fallback: si Supabase no tiene datos, usar comentarios en memoria
    if not context and df_fallback is not None and not df_fallback.empty:
        comments = df_fallback.get("comment", pd.Series())
        sample = comments.dropna().astype(str).head(20).tolist()
        if sample:
            context = "\n---\n".join(sample)

    # Info factual para preguntas de conteo
    stats_text = ""
    if stats:
        total = sum(stats.values())
        stats_text = f"\nEstadisticas de la sesion: {total} comentarios totales.\n"
        for p, c in stats.items():
            stats_text += f"- {p}: {c} comentarios\n"

    history_text = ""
    for msg in chat_history[-6:]:
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""{stats_text}
Contexto de comentarios recolectados:
---
{context if context else "No se encontro contexto en los comentarios."}
---

Historial: {history_text if history_text else "(nuevo)"}

Pregunta: {question}

Basate en el contexto y las estadisticas. Si la pregunta es sobre cantidades, usa las estadisticas. Responde en espanol, se conciso."""

    return get_llm_response(prompt=prompt, model=model,
                            system_prompt="Eres ChismesitoGPT, analista de redes sociales. Usa las estadisticas para preguntas de cantidades. Responde en espanol.")
