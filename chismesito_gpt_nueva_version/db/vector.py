# -*- coding: utf-8 -*-
"""db/vector.py — Operaciones pgvector en schema chismesito_gpt."""

import logging
from db.supabase_client import get_supabase_client, SCHEMA

logger = logging.getLogger(__name__)
TABLE = "unified_comments"


def insert_embedding(comment_id: str, embedding: list[float]):
    get_supabase_client().schema(SCHEMA).table(TABLE).update(
        {"embedding": embedding}).eq("id", comment_id).execute()


def batch_insert_embeddings(id_embedding_pairs: list[tuple[str, list[float]]]):
    for comment_id, embedding in id_embedding_pairs:
        try:
            get_supabase_client().schema(SCHEMA).table(TABLE).update(
                {"embedding": embedding}).eq("id", comment_id).execute()
        except Exception as e:
            logger.error(f"Embedding para {comment_id}: {e}")


def semantic_search(query_embedding: list[float], session_id: str,
                    user_id: str, top_k: int = 10) -> list[dict]:
    try:
        result = get_supabase_client().schema(SCHEMA).rpc(
            "match_comments",
            {
                "query_embedding": query_embedding,
                "p_session_id": session_id,
                "p_user_id": user_id,
                "match_count": top_k,
            },
        ).execute()
        return result.data if result.data else []
    except Exception as e:
        logger.error(f"match_comments RPC: {e}")
        return []
