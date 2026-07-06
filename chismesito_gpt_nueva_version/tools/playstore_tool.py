# -*- coding: utf-8 -*-
"""tools/playstore_tool.py — Google Play Store scraper."""

import logging
import re
from langchain.tools import tool

logger = logging.getLogger(__name__)

try:
    from google_play_scraper import reviews, Sort, search as play_search, app as play_app
except ImportError:
    reviews = Sort = play_search = play_app = None

from config import SERPAPI_API_KEY
try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None


def _extract_app_id(url_or_query: str) -> str | None:
    match = re.search(r"id=([a-zA-Z0-9._]+)", url_or_query)
    return match.group(1) if match else None


# ─── Resolución de ligas de apps (Google Play) ──────────────────────────────
# Orden de prioridad para encontrar los links/appId de una consulta textual:
#   1. Apify google-search-scraper (DEFAULT — $2.5/1000, robusto)
#   2. SerpAPI (EMERGENCIA — 250 consultas/mes gratis)
#   3. google-play-scraper search local (ÚLTIMO RECURSO — gratis, puede fallar)

_GOOGLE_SEARCH_ACTOR = "apify/google-search-scraper"


def _links_from_apify_google(query: str, max_items: int) -> list[dict]:
    """DEFAULT: usa el actor de búsqueda de Google en Apify para obtener appIds."""
    try:
        from tools.apify_tool import _run_actor
        from config import APIFY_API_KEY
    except Exception:
        return []
    if not APIFY_API_KEY:
        return []
    run_input = {
        "queries": f"site:play.google.com/store/apps/details {query}",
        "resultsPerPage": max(10, min(max_items * 2, 30)),
        "maxPagesPerQuery": 1,
        "countryCode": "mx",
        "languageCode": "es",
    }
    try:
        pages = _run_actor(_GOOGLE_SEARCH_ACTOR, run_input, timeout=120)
    except Exception as e:
        logger.warning(f"PlayStore: Apify google-search-scraper falló: {e}")
        return []

    out, seen = [], set()
    for page in pages or []:
        for r in page.get("organicResults", []) or []:
            link = r.get("url", "")
            m = re.search(r"id=([a-zA-Z0-9._]+)", link)
            if not m:
                continue
            app_id = m.group(1)
            if app_id in seen:
                continue
            seen.add(app_id)
            title = re.sub(r"\s*-\s*(Apps on Google Play|Aplicaciones en Google Play).*$", "",
                           r.get("title", "")).strip()
            out.append({"appId": app_id, "title": title or app_id,
                        "url": f"https://play.google.com/store/apps/details?id={app_id}"})
    logger.info(f"PlayStore: Apify Google search → {len(out)} appIds para '{query}'")
    return out


def _links_from_serpapi(query: str, max_items: int) -> list[dict]:
    """EMERGENCIA: SerpAPI (Google Search)."""
    if not (SERPAPI_API_KEY and GoogleSearch is not None):
        return []
    try:
        logger.info(f"PlayStore: EMERGENCIA SerpAPI para: '{query}'")
        params = {
            "q": f"site:play.google.com/store/apps/details {query}",
            "api_key": SERPAPI_API_KEY, "num": max(3, max_items), "engine": "google",
        }
        results = GoogleSearch(params).get_dict()
    except Exception as e:
        logger.warning(f"PlayStore: Error SerpAPI: {e}")
        return []
    out, seen = [], set()
    for r in results.get("organic_results", []):
        link = r.get("link", "")
        m = re.search(r"id=([a-zA-Z0-9._]+)", link)
        if not m or m.group(1) in seen:
            continue
        app_id = m.group(1)
        seen.add(app_id)
        title = re.sub(r"\s*-\s*(Apps on Google Play|Aplicaciones en Google Play).*$", "",
                       r.get("title", "")).strip()
        out.append({"appId": app_id, "title": title or app_id,
                    "url": f"https://play.google.com/store/apps/details?id={app_id}"})
    return out


def _links_from_play_scraper(query: str, max_items: int) -> list[dict]:
    """ÚLTIMO RECURSO: google-play-scraper search (gratis, scraping directo)."""
    if play_search is None:
        return []
    try:
        logger.info(f"PlayStore: ÚLTIMO RECURSO google-play-scraper para: '{query}'")
        try:
            res = play_search(query, lang="es", country="mx", n_hits=max(1, min(max_items, 20)))
        except TypeError:
            res = play_search(query, lang="es", country="mx")
    except Exception as e:
        logger.warning(f"PlayStore: Error google-play-scraper search: {e}")
        return []
    out = []
    for r in (res or [])[:max_items]:
        app_id = r.get("appId")
        if not app_id:
            continue
        out.append({"appId": app_id, "title": r.get("title", app_id),
                    "url": f"https://play.google.com/store/apps/details?id={app_id}",
                    "author": r.get("developer", ""), "icon": r.get("icon", ""),
                    "score": r.get("score")})
    return out


def _links_from_duckduckgo(query: str, max_items: int) -> list[dict]:
    """ÚLTIMO RECURSO: DuckDuckGo (gratis, sin API key)."""
    try:
        from tools.apify_tool import _duckduckgo_search
    except Exception:
        return []
    results = _duckduckgo_search(f"site:play.google.com/store/apps/details {query}", max_items * 2)
    out, seen = [], set()
    for r in results:
        m = re.search(r"id=([a-zA-Z0-9._]+)", r.get("url", ""))
        if not m or m.group(1) in seen:
            continue
        app_id = m.group(1)
        seen.add(app_id)
        title = re.sub(r"\s*-\s*(Apps on Google Play|Aplicaciones en Google Play).*$", "",
                       r.get("title", "")).strip()
        out.append({"appId": app_id, "title": title or app_id,
                    "url": f"https://play.google.com/store/apps/details?id={app_id}"})
    return out


def _resolve_app_links(query: str, max_items: int = 6) -> list[dict]:
    """Cadena: Apify (default) → SerpAPI (emergencia) → scraper local → DuckDuckGo (último recurso)."""
    links = _links_from_apify_google(query, max_items)
    if not links:
        links = _links_from_serpapi(query, max_items)
    if not links:
        links = _links_from_play_scraper(query, max_items)
    if not links:
        links = _links_from_duckduckgo(query, max_items)
    return links[:max_items]


def resolve_playstore_app_id(query: str) -> str | None:
    """Resuelve una consulta textual al ID de paquete oficial de Google Play Store."""
    query = query.strip()
    if not query:
        return None

    # 1. Detectar si ya es un ID de app o contiene un ID de app en URL
    if "details?id=" in query:
        match = re.search(r"id=([a-zA-Z0-9._]+)", query)
        if match:
            return match.group(1)
    if "." in query and " " not in query:
        return query

    # 2. Cadena de resolución (Apify default → SerpAPI → scraper local)
    links = _resolve_app_links(query, max_items=3)
    if links:
        logger.info(f"PlayStore: AppId resuelto: {links[0]['appId']}")
        return links[0]["appId"]
    return None


# ─── FASE 1: Descubrir apps ──────────────────────────────────────────────────

def _enrich_app_card(link: dict) -> dict:
    """Completa metadata (icono, desarrollador, rating) de una app, best-effort."""
    app_id = link["appId"]
    author = link.get("author", "")
    icon = link.get("icon", "")
    score = link.get("score")
    title = link.get("title", app_id)
    # Enriquecer con google-play-scraper app() (gratis y más fiable que su search()).
    if play_app is not None and (not icon or score is None):
        try:
            info = play_app(app_id, lang="es", country="mx")
            title = info.get("title", title)
            author = info.get("developer", author)
            icon = info.get("icon", icon)
            score = info.get("score", score)
        except Exception as e:
            logger.debug(f"PlayStore: no se pudo enriquecer {app_id}: {e}")
    stat = f"{score:.1f} ★" if isinstance(score, (int, float)) else ""
    return {
        "platform": "playstore", "id": app_id, "title": title,
        "author": author or "", "thumbnail": icon or "", "stat": stat,
        "url": link["url"],
    }


def discover_playstore_apps(query: str, max_items: int = 6) -> list[dict]:
    """
    Busca apps en Google Play que coincidan con la consulta y devuelve Items
    normalizados para que el usuario elija de cuáles extraer reviews (Fase 1).

    Ligas vía Apify google-search-scraper (default), SerpAPI (emergencia) o
    google-play-scraper (último recurso). La metadata para las tarjetas se
    completa con google-play-scraper app().

    Returns:
        list[Item] con {platform, id (appId), title, author, thumbnail, stat, url}
    """
    query = (query or "").strip()
    if not query:
        return []

    # Si ya es un appId o URL con id, devolver esa única app (enriquecida).
    direct_id = _extract_app_id(query) if "id=" in query else (
        query if ("." in query and " " not in query) else None
    )
    if direct_id:
        return [_enrich_app_card({"appId": direct_id, "title": direct_id,
                                  "url": f"https://play.google.com/store/apps/details?id={direct_id}"})]

    links = _resolve_app_links(query, max_items=max_items)
    cards = [_enrich_app_card(l) for l in links]
    logger.info(f"PlayStore discover: {len(cards)} apps para '{query}'")
    return cards


# ─── FASE 2: Extraer reviews de apps elegidas ────────────────────────────────

def fetch_playstore_reviews_for(app_ids: list[str], max_comments: int = 50) -> list[dict]:
    """Extrae reviews de una lista de appIds, acumulando hasta `max_comments`."""
    if reviews is None or not app_ids:
        return []
    per_app = max(1, -(-max_comments // len(app_ids)))  # techo de la división
    comments = []
    for app_id in app_ids:
        if len(comments) >= max_comments:
            break
        try:
            reviews_data, _ = reviews(
                app_id, lang="es", country="mx", sort=Sort.NEWEST,
                count=min(per_app, max_comments - len(comments)),
            )
        except Exception as e:
            logger.warning(f"PlayStore reviews error en {app_id}: {e}")
            continue
        for r in reviews_data:
            comments.append({
                "comment": r.get("content", ""),
                "username": r.get("userName", ""),
                "rating": r.get("score", 0),
                "likes": r.get("thumbsUpCount", 0),
                "post_date": r.get("at", ""),
                "social_media": "playstore",
                "url": f"https://play.google.com/store/apps/details?id={app_id}",
            })
            if len(comments) >= max_comments:
                break
    logger.info(f"PlayStore: {len(comments)} reviews de {len(app_ids)} app(s)")
    return comments


@tool
def get_playstore_reviews(query: str, max_results: int = 50) -> dict:
    """
    Obtiene reviews de Google Play Store para una app.

    Args:
        query: ID de la app (ej: 'com.whatsapp'), URL de Play Store o consulta textual (ej: 'Banco Azteca')
        max_results: Maximo de reviews (default 50)

    Returns:
        {"success": bool, "data": [{"comment": str, "username": str, "rating": int, ...}], "count": int, "error": str}
    """
    if reviews is None:
        return {"success": False, "data": [], "count": 0, "error": "google-play-scraper no instalado"}

    try:
        # Resolver query a appId real
        app_id = resolve_playstore_app_id(query)
        if not app_id:
            return {"success": False, "data": [], "count": 0, "error": f"No se pudo resolver el App ID para la consulta: '{query}'"}

        logger.info(f"PlayStore: Extrayendo reviews de la app: '{app_id}'")
        comments = fetch_playstore_reviews_for([app_id], max_comments=max_results)
        return {"success": True, "data": comments, "count": len(comments), "error": None}

    except Exception as e:
        logger.error(f"PlayStore error: {e}")
        return {"success": False, "data": [], "count": 0, "error": str(e)}
