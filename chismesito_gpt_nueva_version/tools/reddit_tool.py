# -*- coding: utf-8 -*-
"""tools/reddit_tool.py — Reddit via PRAW."""

import logging
from langchain.tools import tool
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

logger = logging.getLogger(__name__)

try:
    import praw
except ImportError:
    praw = None


def _client():
    """Devuelve un cliente PRAW o None si faltan credenciales / librería."""
    if praw is None or not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET]):
        return None
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )


# ─── FASE 1: Descubrir posts ─────────────────────────────────────────────────

def discover_reddit_posts(query: str, max_items: int = 8) -> list[dict]:
    """
    Busca posts (submissions) en Reddit y devuelve Items normalizados para que
    el usuario elija de cuáles extraer comentarios (Fase 1).

    Returns:
        list[Item] con {platform, id (submission.id), title, author, thumbnail, stat, url}
    """
    reddit = _client()
    if reddit is None:
        logger.warning("Reddit: sin credenciales/librería; discover vacío")
        return []

    items = []
    try:
        for sub in reddit.subreddit("all").search(query, limit=max(1, min(max_items, 20))):
            thumb = sub.thumbnail if isinstance(getattr(sub, "thumbnail", ""), str) else ""
            if not str(thumb).startswith("http"):
                thumb = ""
            stat = f"{sub.score:,} pts · {sub.num_comments:,} 💬"
            items.append({
                "platform": "reddit",
                "id": sub.id,
                "title": sub.title or "(sin título)",
                "author": f"r/{sub.subreddit.display_name}",
                "thumbnail": thumb,
                "stat": stat,
                "url": f"https://reddit.com{sub.permalink}",
            })
    except Exception as e:
        logger.error(f"Reddit discover error: {e}")
        return []

    logger.info(f"Reddit discover: {len(items)} posts para '{query}'")
    return items


# ─── FASE 2: Extraer comentarios de posts elegidos ──────────────────────────

def fetch_reddit_comments_for(submission_ids: list[str], max_comments: int = 50) -> list[dict]:
    """Extrae comentarios de una lista de submissions, acumulando hasta `max_comments`."""
    reddit = _client()
    if reddit is None or not submission_ids:
        return []

    comments = []
    for sid in submission_ids:
        if len(comments) >= max_comments:
            break
        try:
            submission = reddit.submission(id=sid)
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                if len(comments) >= max_comments:
                    break
                comments.append({
                    "comment": comment.body,
                    "username": str(comment.author) if comment.author else "[deleted]",
                    "likes": comment.score,
                    "post_date": comment.created_utc,
                    "social_media": "reddit",
                    "url": f"https://reddit.com{comment.permalink}",
                })
        except Exception as e:
            logger.warning(f"Reddit comments error en {sid}: {e}")
            continue

    logger.info(f"Reddit: {len(comments)} comentarios de {len(submission_ids)} post(s)")
    return comments


@tool
def get_reddit_comments(query: str, max_results: int = 50) -> dict:
    """
    Obtiene comentarios de Reddit relacionados con una busqueda.

    Args:
        query: Termino de busqueda (ej: 'iPhone 16 review')
        max_results: Maximo de comentarios (default 50)

    Returns:
        {"success": bool, "data": [{"comment": str, "username": str, ...}], "count": int, "error": str}
    """
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET]):
        return {"success": False, "data": [], "count": 0, "error": "Credenciales Reddit no configuradas"}
    if praw is None:
        return {"success": False, "data": [], "count": 0, "error": "praw no instalado"}

    try:
        posts = discover_reddit_posts(query, max_items=5)
        if not posts:
            return {"success": True, "data": [], "count": 0, "error": None}
        comments = fetch_reddit_comments_for([p["id"] for p in posts], max_comments=max_results)
        return {"success": True, "data": comments, "count": len(comments), "error": None}
    except Exception as e:
        logger.error(f"Reddit error: {e}")
        return {"success": False, "data": [], "count": 0, "error": str(e)}
