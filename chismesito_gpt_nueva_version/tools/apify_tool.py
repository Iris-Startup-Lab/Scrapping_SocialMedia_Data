# -*- coding: utf-8 -*-
"""tools/apify_tool.py — APIFY actors (FB, X, TikTok, Instagram, Maps)."""
# Aquí están los métodos principales para Apify que es la herramienta que usamos para obtener comentarios de redes sociales complicadas

import logging, time, re, unicodedata
from langchain.tools import tool
from apify_client import ApifyClient
from config import APIFY_API_KEY, MAPS_API_KEY

logger = logging.getLogger(__name__)

# ─── Config de actors (IDs de actors oficiales provistos por el usuario) ───
ACTOR_MAP: dict[str, dict] = {
    "facebook": {
        "search": "Us34x9p7VgjCz99H6",       # apify/facebook-search-scraper
        "post": "KoJrdxJCTtpon81KY",         # apify/facebook-posts-scraper
        "comments": "us5srxAYnsrkgUv2v",     # apify/facebook-comments-scraper
        "two_step": True,
    },
    "x_twitter": {
        "search": "8CiMefkv2yLlD7vYl",       # watcher.data/search-x-by-keywords
        "comments": "qhybbvlFivx7AP0Oh",     # scraper_one/x-post-replies-scraper
        "two_step": True,
    },
    "twitter": {
        "search": "8CiMefkv2yLlD7vYl",
        "comments": "qhybbvlFivx7AP0Oh",
        "two_step": True,
    },
    "instagram": {
        "search": "reGe1ST3OBgYZSsZJ",       # apify/instagram-hashtag-scraper
        "comments": "SbK00X0JYCPblD2wp",     # apify/instagram-comment-scraper
        "two_step": True,
    },
    "tiktok": {
        "search": "f1ZeP0K58iwlqG2pY",       # clockworks/tiktok-hashtag-scraper
        "comments": "BDec00yAmCm1QbMEI",     # clockworks/tiktok-comments-scraper
        "two_step": True,
    },
    "google_maps": {
        "comments": "compass/Google-Maps-Reviews-Scraper",
        "two_step": False,
    },
    "maps": {
        "comments": "compass/Google-Maps-Reviews-Scraper",
        "two_step": False,
    },
}

# Nombres de plataformas para parseo
PLATFORM_LABELS = {
    "facebook": "facebook", "x_twitter": "twitter", "twitter": "twitter",
    "instagram": "instagram", "tiktok": "tiktok",
    "google_maps": "maps", "maps": "maps",
}


def _run_actor(actor_id: str, input_data: dict, timeout: int = 180) -> list[dict]:
    """Ejecuta un actor de Apify utilizando la librería oficial apify-client."""
    client = ApifyClient(APIFY_API_KEY)
    logger.info(f"APIFY: Iniciando ejecución de actor '{actor_id}'")
    try:
        from datetime import timedelta
        run = client.actor(actor_id).call(run_input=input_data, wait_duration=timedelta(seconds=timeout), logger=None)
        if not run:
            logger.warning(f"APIFY: El actor '{actor_id}' retornó None.")
            return []
        
        # Obtener dataset_id de forma segura compatible con Pydantic y dicts
        dataset_id = None
        if hasattr(run, "default_dataset_id"):
            dataset_id = run.default_dataset_id
        elif hasattr(run, "get"):
            dataset_id = run.get("default_dataset_id") or run.get("defaultDatasetId")

        if not dataset_id:
            logger.warning(f"APIFY: El actor '{actor_id}' no retornó dataset_id.")
            return []
        
        items = list(client.dataset(dataset_id).iterate_items())
        logger.info(f"APIFY: Actor '{actor_id}' completado, {len(items)} ítems obtenidos.")
        # Registrar costo APIFY (pago por resultado) en el tracker activo.
        try:
            from utils.cost_tracker import record_apify
            record_apify(actor_id, len(items))
        except Exception as e:
            logger.debug(f"No se pudo registrar costo APIFY: {e}")
        return items
    except Exception as e:
        logger.error(f"APIFY: Error ejecutando actor '{actor_id}': {e}")
        return []


def _extract_clean_keyword(query: str) -> tuple[str, bool]:
    """
    Llama al LLM para extraer la entidad o palabra clave principal de un prompt de búsqueda.
    Útil para hashtags/keywords limpios.
    Retorna una tupla: (keyword, gemma_used: bool)
    """
    q = query.strip()
    if not q:
        return q, False
    if q.startswith("#"):
        return q[1:].strip(), False
    if " " not in q and "." not in q and "http" not in q:
        return q, False
    if q.startswith("http://") or q.startswith("https://") or q.startswith("place_id:"):
        return q, False

    try:
        from llm_manager import get_llm_response
        system_prompt = (
            "Eres un extractor de términos de búsqueda. Tu única tarea es extraer la entidad, "
            "persona, marca, producto, tema o concepto principal del prompt del usuario para usarlo "
            "como palabra clave de búsqueda. Responde ÚNICAMENTE con el término extraído, "
            "sin explicaciones, sin comillas, sin signos de puntuación y sin el símbolo #."
        )
        # Usamos gemma-4-31b-it para la interpretación del hashtag/keyword
        extracted = get_llm_response(f"Prompt: '{q}'", model="gemma-4-31b-it", system_prompt=system_prompt)
        extracted = extracted.strip().strip("'\"#.")
        if extracted and len(extracted) < len(q):
            logger.info(f"LLM keyword extraction: '{q}' -> '{extracted}'")
            return extracted, True
    except Exception as e:
        logger.warning(f"Error al extraer keyword con LLM: {e}")
    return q, False


def _extract_comments(items: list[dict], platform: str) -> list[dict]:
    """Extrae campos comunes de comentarios de items APIFY."""
    label = PLATFORM_LABELS.get(platform, platform)
    comments = []
    for item in items:
        # Extraer texto de comentario
        comm_text = item.get("text", item.get("comment", item.get("reviewText", item.get("replyText", item.get("reply", "")))))
        if not comm_text:
            continue

        comments.append({
            "comment": comm_text,
            "username": item.get("username", item.get("author", item.get("name", item.get("screenName", item.get("owner", ""))))),
            "url": item.get("url", item.get("replyUrl", "")),
            "likes": item.get("likes", item.get("likeCount", item.get("thumbsUpCount", 0))),
            "post_date": item.get("date", item.get("timestamp", item.get("publishDate", ""))),
            "rating": item.get("stars", item.get("rating", item.get("score", None))),
            "social_media": label,
        })
    return comments


# ─── Acumulación por lotes de posts hasta alcanzar el número solicitado ───
# Se procesan los posts en lotes; tras cada corrida se deduplica y se sigue
# con el siguiente lote hasta llegar a max_items o agotar posts/corridas.
COMMENTS_BATCH_SIZE = 3    # posts por corrida del actor de comentarios
MAX_COMMENT_RUNS = 5       # tope de corridas para acotar el costo de APIFY


def _comments_input_for(platform: str, urls: list[str], want: int) -> dict:
    """Construye el input del actor de comentarios para un lote de URLs."""
    if platform == "instagram":
        return {"directUrls": urls, "resultsLimit": want}
    elif platform == "tiktok":
        return {
            "postURLs": urls,
            "commentsPerPost": want,
            "maxRepliesPerComment": 0,
            "resultsPerPage": want,
            "profileScrapeSections": ["videos"],
            "profileSorting": "latest",
            "excludePinnedPosts": False,
        }
    elif platform in ("twitter", "x_twitter"):
        return {"postUrls": urls, "resultsLimit": want}
    else:  # facebook
        return {
            "startUrls": [{"url": u} for u in urls],
            "resultsLimit": want,
            "includeNestedComments": False,
            "viewOption": "RANKED_UNFILTERED",
        }


def _collect_comments_batched(actor_id: str, platform: str, urls: list[str],
                              max_items: int, timeout: int = 150) -> list[dict]:
    """
    Recorre los posts en lotes, acumulando comentarios deduplicados hasta llegar
    a `max_items` o agotar los posts / el tope de corridas. Así, si un post
    trae pocos comentarios, se compensa con los siguientes posts.
    """
    collected: list[dict] = []
    seen: set[str] = set()
    runs = 0
    for i in range(0, len(urls), COMMENTS_BATCH_SIZE):
        if len(collected) >= max_items or runs >= MAX_COMMENT_RUNS:
            break
        batch = urls[i:i + COMMENTS_BATCH_SIZE]
        # Pedimos el objetivo completo por corrida; el dedup descarta repetidos.
        comments_input = _comments_input_for(platform, batch, max_items)
        try:
            items = _run_actor(actor_id, comments_input, timeout=timeout)
        except Exception as e:
            logger.error(f"APIFY {platform} comments batch error: {e}")
            items = []
        runs += 1
        for c in _extract_comments(items, platform):
            txt = (c.get("comment") or "").strip()
            if txt and txt not in seen:
                seen.add(txt)
                collected.append(c)
                if len(collected) >= max_items:
                    break
    if len(collected) < max_items:
        logger.info(
            f"APIFY {platform}: {len(collected)}/{max_items} comentarios "
            f"tras {runs} corrida(s) (posts agotados o límite de corridas)"
        )
    return collected[:max_items]


# ─── FASE 1: Descubrir posts/videos (solo el paso de búsqueda) ───────────────

def _search_item_to_card(item: dict, platform: str, url: str) -> dict:
    """Normaliza un item de búsqueda APIFY a un Item de tarjeta para la UI."""
    text = (item.get("text") or item.get("caption") or item.get("title")
            or item.get("message") or item.get("description") or "").strip()
    title = (text[:120] + "…") if len(text) > 120 else (text or "(sin texto)")
    author = (item.get("username") or item.get("author") or item.get("ownerUsername")
              or item.get("authorMeta", {}).get("name") if isinstance(item.get("authorMeta"), dict)
              else item.get("username") or item.get("author") or "")
    if isinstance(author, dict):
        author = author.get("name", "")
    likes = (item.get("likes") or item.get("likesCount") or item.get("likeCount")
             or item.get("diggCount") or item.get("playCount") or 0)
    stat = f"{int(likes):,} 👍" if isinstance(likes, (int, float)) and likes else ""
    thumb = (item.get("thumbnailUrl") or item.get("displayUrl") or item.get("thumbnail")
             or item.get("videoThumbnail") or item.get("previewImage") or "")
    return {
        "platform": PLATFORM_LABELS.get(platform, platform),
        "id": url,             # para APIFY el handle de la Fase 2 es la URL del post
        "title": title,
        "author": str(author or ""),
        "thumbnail": thumb,
        "stat": stat,
        "url": url,
    }


# ─── Búsqueda barata de posts vía Google (Apify google-search-scraper) ───────
# El facebook-search-scraper ($12/1000) + facebook-posts-scraper ($4/1000) son
# caros; los reemplazamos por google-search-scraper ($2.5/1000) con site: y
# pasamos las URLs directo al comments-scraper. Fallback al método clásico.
_GOOGLE_SEARCH_ACTOR = "apify/google-search-scraper"
_FB_POST_PATTERNS = ("/posts/", "/permalink", "story_fbid", "/videos/", "/photos/", "/reel/")


def _apify_google_search(full_query: str, max_results: int) -> list[dict]:
    """Ejecuta el actor de búsqueda de Google en Apify. Devuelve [{url,title}]."""
    if not APIFY_API_KEY:
        return []
    run_input = {
        "queries": full_query,
        "resultsPerPage": max(10, min(max_results, 50)),
        "maxPagesPerQuery": 1,
        "countryCode": "mx",
        "languageCode": "es",
    }
    try:
        pages = _run_actor(_GOOGLE_SEARCH_ACTOR, run_input, timeout=120)
    except Exception as e:
        logger.warning(f"APIFY google-search-scraper falló: {e}")
        return []
    out = []
    for page in pages or []:
        for r in page.get("organicResults", []) or []:
            if r.get("url"):
                out.append({"url": r["url"], "title": r.get("title", "")})
    return out


def _duckduckgo_search(full_query: str, max_results: int) -> list[dict]:
    """ÚLTIMO RECURSO gratis (sin API key): DuckDuckGo. Devuelve [{url,title}].

    Puede ser inestable/rate-limited; por eso solo se usa como último fallback.
    Requiere `pip install ddgs` (o el antiguo `duckduckgo_search`).
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("DuckDuckGo: instala 'ddgs' (pip install ddgs) para el fallback")
        return []
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(full_query, region="mx-es", max_results=max(5, min(max_results, 30))):
                url = r.get("href") or r.get("url") or ""
                if url:
                    out.append({"url": url, "title": r.get("title", "")})
    except Exception as e:
        logger.warning(f"DuckDuckGo search falló: {e}")
    logger.info(f"DuckDuckGo: {len(out)} resultados para '{full_query}'")
    return out


def _facebook_cards_from_results(results: list[dict], max_items: int) -> list[dict]:
    """Filtra resultados de búsqueda a URLs de post de FB y arma tarjetas."""
    cards, seen = [], set()
    for r in results:
        url = r.get("url", "")
        if not any(pat in url for pat in _FB_POST_PATTERNS) or url in seen:
            continue
        seen.add(url)
        title = (r.get("title") or url).replace(" | Facebook", "").strip()
        cards.append({
            "platform": "facebook", "id": url,
            "title": title[:120], "author": "", "thumbnail": "", "stat": "", "url": url,
        })
        if len(cards) >= max_items:
            break
    return cards


def _facebook_discover_via_google(kw: str, max_items: int) -> list[dict]:
    """Descubre posts de Facebook con Google (barato) en vez de los actores caros."""
    cards = _facebook_cards_from_results(_apify_google_search(f"site:facebook.com {kw}", max_items * 3), max_items)
    logger.info(f"Facebook: Google search → {len(cards)} posts (barato, $2.5/1000)")
    return cards


def _facebook_discover_via_ddg(kw: str, max_items: int) -> list[dict]:
    """Descubre posts de Facebook con DuckDuckGo (gratis, último recurso)."""
    cards = _facebook_cards_from_results(_duckduckgo_search(f"site:facebook.com {kw}", max_items * 3), max_items)
    logger.info(f"Facebook: DuckDuckGo → {len(cards)} posts (gratis)")
    return cards


def apify_discover(query: str, platform: str, max_items: int = 10, keyword: str | None = None) -> list[dict]:
    """
    Ejecuta SOLO el paso de búsqueda de APIFY y devuelve los posts/videos
    encontrados como Items normalizados (Fase 1 del flujo manual).
    Facebook usa Google search (barato); el resto usa el actor de búsqueda propio.

    `keyword`: palabra clave ya extraída; si es None se extrae con el LLM
    (evita una segunda llamada al LLM cuando el llamador ya la calculó).
    """
    if not APIFY_API_KEY:
        return []
    platform = platform.lower().strip()
    if platform not in ACTOR_MAP or not ACTOR_MAP[platform].get("two_step"):
        return []

    cfg = ACTOR_MAP[platform]
    kw = keyword if keyword is not None else _extract_clean_keyword(query)[0]
    search_limit = max(max_items, COMMENTS_BATCH_SIZE * MAX_COMMENT_RUNS)

    # Facebook: rutas baratas primero (Google $2.5/1000 → DuckDuckGo gratis);
    # los actores caros ($12+$4/1000) solo como último recurso.
    if platform == "facebook":
        cards = _facebook_discover_via_google(kw, max_items)
        if not cards:
            cards = _facebook_discover_via_ddg(kw, max_items)
        if cards:
            return cards
        logger.info("Facebook: Google/DDG sin posts; uso facebook-search-scraper (CARO $12+$4/1000)")

    search_input = _build_search_input(platform, kw, search_limit)

    search_items = _run_actor(cfg["search"], search_input, timeout=180)
    if not search_items:
        return []

    # Emparejar cada URL con su item de búsqueda para conservar la metadata.
    pairs: list[tuple[str, dict]] = []
    for item in search_items[:search_limit]:
        url = _extract_post_url(item, platform)
        if url:
            pairs.append((url, item))

    # Facebook: resolver URLs reales de posts a partir de las páginas.
    if platform == "facebook" and pairs:
        start_urls = [{"url": u} for u, _ in pairs[:search_limit]]
        try:
            post_items = _run_actor(cfg["post"], {
                "startUrls": start_urls, "resultsLimit": search_limit, "captionText": False,
            }, timeout=120)
        except Exception as e:
            logger.error(f"APIFY facebook posts scraper error: {e}")
            post_items = []
        pairs = []
        for item in post_items:
            u = item.get("url", item.get("postUrl", item.get("link", item.get("facebookUrl", ""))))
            if u:
                pairs.append((u, item))

    cards = [_search_item_to_card(item, platform, url) for url, item in pairs]
    logger.info(f"APIFY discover {platform}: {len(cards)} posts para '{query}'")
    return cards


def apify_fetch_comments(platform: str, urls: list[str], max_items: int = 10) -> list[dict]:
    """
    Extrae comentarios de una lista de URLs de posts ya elegidas (Fase 2).
    """
    if not APIFY_API_KEY or not urls:
        return []
    platform = platform.lower().strip()
    if platform not in ACTOR_MAP:
        return []
    max_items = min(max_items, 50)
    return _collect_comments_batched(ACTOR_MAP[platform]["comments"], platform, urls, max_items)


def _extract_post_url(item: dict, platform: str) -> str:
    """Extrae la URL de un item de búsqueda, normalizando IDs sueltos."""
    url = item.get("url", item.get("postUrl", item.get("webVideoUrl",
          item.get("link", item.get("pageUrl", item.get("id", ""))))))
    if not url:
        return ""
    if not str(url).startswith("http"):
        if platform == "instagram":
            url = f"https://www.instagram.com/p/{url}/"
        elif platform == "tiktok":
            url = f"https://www.tiktok.com/@tiktok/video/{url}"
    return url


def _build_search_input(platform: str, kw: str, search_limit: int) -> dict:
    """Construye el input del actor de búsqueda (Paso 1) según la plataforma."""
    if platform == "instagram":
        clean_kw = unicodedata.normalize('NFKD', kw).encode('ascii', 'ignore').decode('ascii')
        hashtag = re.sub(r'[^a-zA-Z0-9]', '', clean_kw).lower()
        logger.info(f"Instagram: Hashtag extraído: '{hashtag}'")
        return {"hashtags": [hashtag], "resultsType": "posts", "resultsLimit": search_limit}
    elif platform == "tiktok":
        clean_kw = unicodedata.normalize('NFKD', kw).encode('ascii', 'ignore').decode('ascii')
        hashtag = re.sub(r'[^a-zA-Z0-9]', '', clean_kw).lower()
        logger.info(f"TikTok: Hashtag extraído: '{hashtag}'")
        return {
            "hashtags": [hashtag], "resultsPerPage": search_limit,
            "shouldDownloadVideos": False, "shouldDownloadCovers": False,
            "downloadSubtitlesOptions": "NEVER_DOWNLOAD_SUBTITLES",
            "shouldDownloadSlideshowImages": False, "videoKvStoreIdOrName": "tiktok-videos",
        }
    elif platform in ("twitter", "x_twitter"):
        logger.info(f"X/Twitter: Palabra clave extraída: '{kw}'")
        return {
            "searchType": "tweets", "keywords": [kw], "maxItemsPerKeyword": search_limit,
            "sortBy": "latest", "outputFormat": "json",
            "proxyConfiguration": {"useApifyProxy": False},
        }
    else:  # facebook
        logger.info(f"Facebook: Palabra clave extraída: '{kw}'")
        return {"searchTerms": [kw], "categories": ["Pub"], "locations": [], "resultsLimit": search_limit}


@tool
def apify_scraper(query: str, platform: str, max_items: int = 10) -> dict:
    """
    Extrae comentarios de redes sociales usando APIFY.
    Facebook, TikTok, Instagram: busca posts y luego extrae comentarios (2 pasos).
    Google Maps: usa la API oficial de Maps primero, APIFY como fallback.
    X/Twitter: busca keywords y extrae replies.

    Args:
        query: Termino de busqueda o prompt
        platform: 'facebook', 'twitter', 'instagram', 'tiktok', 'google_maps'
        max_items: Maximo de items (default 10)
    """
    if not APIFY_API_KEY:
        return {"success": False, "data": [], "count": 0, "error": "APIFY_API_KEY no configurada", "gemma_processed": False}

    platform = platform.lower().strip()
    if platform not in ACTOR_MAP:
        return {"success": False, "data": [], "count": 0, "error": f"Plataforma '{platform}' no soportada", "gemma_processed": False}

    cfg = ACTOR_MAP[platform]
    max_items = min(max_items, 50)
    gemma_used = False

    try:
        if cfg["two_step"]:
            # Paso 1: buscar posts / hashtags (reusa apify_discover para no duplicar).
            logger.info(f"APIFY {platform} — Buscando contenido para: '{query}'...")

            # Extraemos la keyword aquí una sola vez y se la pasamos a apify_discover
            # (evita una segunda llamada al LLM).
            kw, gemma_used = _extract_clean_keyword(query)
            cards = apify_discover(query, platform, max_items, keyword=kw)

            if not cards:
                return {"success": True, "data": [], "count": 0, "error": None, "gemma_processed": gemma_used}

            urls = [c["id"] for c in cards]
            logger.info(f"APIFY {platform} — extrayendo comentarios de hasta {len(urls)} posts...")
            all_comments = apify_fetch_comments(platform, urls, max_items)
            return {"success": True, "data": all_comments,
                    "count": len(all_comments), "error": None, "gemma_processed": gemma_used}
        else:
            # Google Maps: 1 solo paso con APIFY
            logger.info(f"APIFY {platform} — extrayendo reviews...")
            # Si el query es un place_id específico
            if query.startswith("place_id:"):
                place_id_val = query.replace("place_id:", "").strip()
                input_data = {
                    "placeIds": [place_id_val],
                    "maxReviews": max_items,
                    "maxItems": max_items,
                }
            elif query.startswith("http://") or query.startswith("https://"):
                input_data = {
                    "startUrls": [{"url": query}],
                    "maxReviews": max_items,
                    "maxItems": max_items,
                }
            else:
                input_data = {
                    "searchStrings": [query],
                    "maxReviews": max_items,
                    "maxItems": max_items,
                }
            items = _run_actor(cfg["comments"], input_data, timeout=180)
            comments = _extract_comments(items, platform)
            return {"success": True, "data": comments[:max_items],
                    "count": len(comments[:max_items]), "error": None, "gemma_processed": False}

    except Exception as e:
        logger.error(f"APIFY {platform} error: {e}")
        return {"success": False, "data": [], "count": 0, "error": str(e), "gemma_processed": False}
