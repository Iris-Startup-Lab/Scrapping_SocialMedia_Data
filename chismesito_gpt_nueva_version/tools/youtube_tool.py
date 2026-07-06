# -*- coding: utf-8 -*-
"""tools/youtube_tool.py — YouTube Data API v3."""

import logging
import re
from langchain.tools import tool
from config import YOUTUBE_API_KEY

logger = logging.getLogger(__name__)

try:
    from googleapiclient.discovery import build
except ImportError:
    build = None


def _extract_video_id(url_or_query: str) -> str | None:
    match = re.search(r"(?:v=|/)([a-zA-Z0-9_-]{11})", url_or_query)
    return match.group(1) if match else None


def _client():
    """Devuelve un cliente de YouTube o None si no hay API key / librería."""
    if not YOUTUBE_API_KEY or build is None:
        return None
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


# ─── FASE 1: Descubrir videos ────────────────────────────────────────────────

def discover_youtube_videos(query: str, max_items: int = 8) -> list[dict]:
    """
    Busca videos relacionados con el query y devuelve items normalizados para
    que el usuario elija de cuáles extraer comentarios (Fase 1 del flujo manual).

    Returns:
        list[Item] con {platform, id (videoId), title, author, thumbnail, stat, url}
    """
    youtube = _client()
    if youtube is None:
        logger.warning("YouTube: sin API key o librería; discover devuelve vacío")
        return []

    # Si el query es una URL/ID directa, devolver ese único video
    vid = _extract_video_id(query)
    try:
        if vid:
            resp = youtube.videos().list(part="snippet,statistics", id=vid).execute()
            items = resp.get("items", [])
        else:
            search = youtube.search().list(
                q=query, part="snippet", type="video", maxResults=max(1, min(max_items, 25))
            ).execute()
            items = search.get("items", [])
    except Exception as e:
        logger.error(f"YouTube discover error: {e}")
        return []

    results = []
    for it in items:
        # search().list → id.videoId ; videos().list → id (str)
        raw_id = it.get("id")
        video_id = raw_id.get("videoId") if isinstance(raw_id, dict) else raw_id
        if not video_id:
            continue
        sn = it.get("snippet", {})
        thumbs = sn.get("thumbnails", {})
        thumb = (thumbs.get("medium") or thumbs.get("default") or {}).get("url", "")
        stats = it.get("statistics", {})
        views = stats.get("viewCount")
        stat = f"{int(views):,} vistas" if views and str(views).isdigit() else ""
        results.append({
            "platform": "youtube",
            "id": video_id,
            "title": sn.get("title", "(sin título)"),
            "author": sn.get("channelTitle", ""),
            "thumbnail": thumb,
            "stat": stat,
            "url": f"https://youtube.com/watch?v={video_id}",
        })
    logger.info(f"YouTube discover: {len(results)} videos para '{query}'")
    return results


# ─── FASE 2: Extraer comentarios de videos elegidos ──────────────────────────

def fetch_youtube_comments(video_ids: list[str], max_comments: int = 50) -> list[dict]:
    """
    Extrae comentarios de una lista de videoIds, acumulando entre videos hasta
    llegar a `max_comments` (Fase 2 del flujo manual).
    """
    youtube = _client()
    if youtube is None or not video_ids:
        return []

    comments = []
    for video_id in video_ids:
        if len(comments) >= max_comments:
            break
        page_token = None
        try:
            while len(comments) < max_comments:
                resp = (
                    youtube.commentThreads()
                    .list(part="snippet", videoId=video_id,
                          maxResults=min(100, max_comments - len(comments)),
                          pageToken=page_token)
                    .execute()
                )
                for item in resp.get("items", []):
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append({
                        "comment": snippet.get("textDisplay", ""),
                        "username": snippet.get("authorDisplayName", ""),
                        "likes": snippet.get("likeCount", 0),
                        "post_date": snippet.get("publishedAt", ""),
                        "social_media": "youtube",
                        "url": f"https://youtube.com/watch?v={video_id}",
                    })
                    if len(comments) >= max_comments:
                        break
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
        except Exception as e:
            logger.warning(f"YouTube comments error en video {video_id}: {e}")
            continue

    logger.info(f"YouTube: {len(comments)} comentarios de {len(video_ids)} video(s)")
    return comments


@tool
def get_youtube_comments(query: str, max_results: int = 50) -> dict:
    """
    Obtiene comentarios de videos de YouTube.

    Busca videos relacionados con el query y extrae comentarios del video mas relevante.

    Args:
        query: Termino de busqueda (ej: 'iPhone 16 review') o URL de video
        max_results: Maximo de comentarios (default 50)

    Returns:
        {"success": bool, "data": [{"comment": str, "username": str, ...}], "count": int, "error": str}
    """
    if not YOUTUBE_API_KEY:
        return {"success": False, "data": [], "count": 0, "error": "YOUTUBE_API_KEY no configurada"}
    if build is None:
        return {"success": False, "data": [], "count": 0, "error": "google-api-python-client no instalado"}

    try:
        # Modo automático: tomar el video más relevante y extraer sus comentarios.
        videos = discover_youtube_videos(query, max_items=1)
        if not videos:
            return {"success": False, "data": [], "count": 0, "error": f"No se encontraron videos para '{query}'"}

        comments = fetch_youtube_comments([videos[0]["id"]], max_comments=max_results)
        return {"success": True, "data": comments, "count": len(comments), "error": None}

    except Exception as e:
        logger.error(f"YouTube error: {e}")
        return {"success": False, "data": [], "count": 0, "error": str(e)}
