# -*- coding: utf-8 -*-
"""db/ops.py — CRUD en unified_comments."""

import logging
import pandas as pd
import numpy as np
import math
from datetime import datetime, timezone
from db.supabase_client import get_supabase_client, SCHEMA

logger = logging.getLogger(__name__)
TABLE = "unified_comments"

def _table():
    return get_supabase_client().schema(SCHEMA).table(TABLE)


def insert_comments(df: pd.DataFrame, user_id: str, session_id: str,
                    query: str, social_media: str) -> int:
    if df.empty:
        return 0
    rows = []
    for _, row in df.iterrows():
        # Sanitizar NaN → None para JSON
        def _safe_float(v):
            try:
                f = float(v)
                return f if not math.isnan(f) else None
            except (ValueError, TypeError):
                return None
        def _safe_int(v):
            try:
                f = float(v)
                return int(v) if not math.isnan(f) else None
            except (ValueError, TypeError):
                return None
        username = row.get("username") or row.get("author")
        rows.append({
            "user_id": user_id, "session_id": session_id,
            "social_media": social_media, "query": query,
            "comment": str(row.get("comment", row.get("text", ""))),
            "username": str(username) if pd.notna(username) and username else None,
            "url": str(row.get("url", "")) if row.get("url") and pd.notna(row.get("url")) else None,
            "rating": _safe_float(row.get("rating")),
            "likes": _safe_int(row.get("likes")),
            "category": str(row.get("category", "")) if row.get("category") and pd.notna(row.get("category")) else None,
            "sentiment": str(row.get("sentiment", "")) if row.get("sentiment") and pd.notna(row.get("sentiment")) else None,
            "emotion": str(row.get("emotion", "")) if row.get("emotion") and pd.notna(row.get("emotion")) else None,
            "extraction_date": datetime.now(timezone.utc).isoformat(),
        })
    logger.info(f"Intentando insertar {len(rows)} filas en {SCHEMA}.{TABLE} (social_media={social_media})")
    try:
        result = _table().insert(rows).execute()
        logger.info(f"Insertadas {len(result.data)} filas en {SCHEMA}.{TABLE}")
        return len(result.data)
    except Exception as e:
        logger.error(f"Error insertando en Supabase: {e}")
        raise


def get_session_comments(session_id: str, user_id: str) -> pd.DataFrame:
    result = _table().select("*").eq("session_id", session_id).eq("user_id", user_id).execute()
    return pd.DataFrame(result.data) if result.data else pd.DataFrame()


def update_comment_analysis(comment_id: str, category=None, sentiment=None, emotion=None):
    updates = {}
    if category is not None: updates["category"] = category
    if sentiment is not None: updates["sentiment"] = sentiment
    if emotion is not None: updates["emotion"] = emotion
    if updates:
        _table().update(updates).eq("id", comment_id).execute()


def insert_user_request(user_id: str, user_email: str, session_id: str,
                        query: str, social_medias: list[str],
                        comments_count: int, model: str, cost: float) -> dict | None:
    """Inserta una petición de búsqueda en la tabla user_requests."""
    try:
        platforms_str = ", ".join(social_medias)
        row = {
            "user_id": user_id,
            "user_email": user_email,
            "session_id": session_id,
            "query": query,
            "social_medias": platforms_str,
            "comments_count": comments_count,
            "model": model,
            "cost": float(cost),
        }
        logger.info(f"Insertando peticion en {SCHEMA}.user_requests para {user_email}")
        result = get_supabase_client().schema(SCHEMA).table("user_requests").insert(row).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Error al insertar en user_requests: {e}")
        return None


def insert_request_performance(session_id: str, execution_time: float,
                               llm_model: str, classification_time: float,
                               embedding_model: str, embedding_time: float,
                               max_comments: int, social_medias: list[str]) -> dict | None:
    """Inserta metricas de rendimiento en la tabla request_performance."""
    try:
        row = {
            "session_id": session_id,
            "execution_time": float(execution_time),
            "llm_model": llm_model,
            "classification_time": float(classification_time),
            "embedding_model": embedding_model,
            "embedding_time": float(embedding_time),
            "max_comments": int(max_comments),
            "num_social_medias": len(social_medias),
            "social_medias": social_medias,
        }
        logger.info(f"Insertando metricas de rendimiento para la sesion {session_id}")
        result = get_supabase_client().schema(SCHEMA).table("request_performance").insert(row).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Error al insertar en request_performance: {e}")
        return None
