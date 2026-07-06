# -*- coding: utf-8 -*-
"""pipeline/orchestrator.py — Orquestador principal: scrape → analyze → store."""

import logging
import uuid
import pandas as pd
import time
from tools.youtube_tool import get_youtube_comments, discover_youtube_videos, fetch_youtube_comments
from tools.reddit_tool import get_reddit_comments, discover_reddit_posts, fetch_reddit_comments_for
from tools.playstore_tool import get_playstore_reviews, discover_playstore_apps, fetch_playstore_reviews_for
from tools.google_maps_tool import get_google_maps_reviews, discover_maps_places, _resolve_coordinates
from tools.apify_tool import apify_scraper, apify_discover, apify_fetch_comments
from tools.search_tool import search_web
from pipeline.analyzer import analyze_comments
from db.ops import insert_comments
from db.vector import batch_insert_embeddings
from db.supabase_client import SCHEMA
from config import MAPS_API_KEY

logger = logging.getLogger(__name__)

# Mapeo: plataforma → (tool_func, usa_apify)
PLATFORM_CONFIG = {
    "youtube":    (get_youtube_comments,    "api"),
    "reddit":     (get_reddit_comments,     "api"),
    "playstore":  (get_playstore_reviews,   "api"),
    "facebook":   (apify_scraper,           "apify"),
    "twitter":    (apify_scraper,           "apify"),
    "x_twitter":  (apify_scraper,           "apify"),
    "instagram":  (apify_scraper,           "apify"),
    "tiktok":     (apify_scraper,           "apify"),
    "maps":        (get_google_maps_reviews, "api"),
    "google_maps": (get_google_maps_reviews, "api"),
}


def run_pipeline(
    prompt: str,
    social_medias: list[str],
    user_id: str,
    session_id: str | None = None,
    model: str = "gemini-2.0-flash-lite",
    max_comments: int = 10,
    maps_geo_toggle: bool = False,
    maps_location: str = "",
    maps_radius: int = 2000,
    user_email: str | None = None,
) -> dict:
    start_time = time.time()
    """
    Pipeline completo: busca, extrae, analiza y guarda.

    Args:
        prompt: La pregunta del usuario
        social_medias: Lista de plataformas seleccionadas
        user_id: ID del usuario
        session_id: ID de sesion (se genera si es None)
        model: Modelo LLM a usar para generacion de categorias
        max_comments: Maximo de comentarios por plataforma (default 10)
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    all_comments = []
    stats = {}
    errors = []
    gemma_processed = False

    for platform in social_medias:
        platform = platform.lower().strip()
        if platform not in PLATFORM_CONFIG:
            errors.append(f"Plataforma no soportada: {platform}")
            continue

        logger.info(f"Procesando {platform}...")

        try:
            # ── Google Maps: API busca lugares → APIFY scrappea cada uno ──
            if platform in ("maps", "google_maps"):
                if not MAPS_API_KEY:
                    errors.append("maps: MAPS_API_KEY no configurada")
                    stats[platform] = 0
                    continue

                import googlemaps
                gmaps = googlemaps.Client(key=MAPS_API_KEY)

                # Búsqueda geográfica o de texto. Tomamos varios lugares para poder
                # acumular comentarios entre sitios hasta alcanzar el objetivo.
                MAX_PLACES = 10
                if maps_geo_toggle:
                    center = _resolve_coordinates(gmaps, maps_location)
                    if not center:
                        errors.append(f"maps: No se pudo resolver la ubicación '{maps_location}'")
                        places_found = []
                    else:
                        logger.info(f"Maps: Búsqueda geográfica en {center} con radio {maps_radius}m")
                        places_resp = gmaps.places_nearby(location=center, radius=maps_radius, keyword=prompt)
                        places_found = places_resp.get("results", [])[:MAX_PLACES]
                else:
                    places_resp = gmaps.places(query=prompt)
                    places_found = places_resp.get("results", [])[:MAX_PLACES]

                if not places_found:
                    # Fallback a APIFY directo sin coordenadas específicas
                    logger.info("Maps Places API sin resultados, probando APIFY directo...")
                    result = apify_scraper.invoke({"query": prompt, "platform": platform, "max_items": max_comments})
                    comments = result.get("data", [])
                    gemma_processed = gemma_processed or result.get("gemma_processed", False)
                    # Sin coordenadas específicas, el mapa no podrá ubicarlas
                    for c in comments:
                        c["latitude"] = None
                        c["longitude"] = None
                else:
                    # Extraer reviews de cada lugar via APIFY, acumulando (con dedup)
                    # entre sitios hasta alcanzar max_comments.
                    place_names = [p.get("name", "") for p in places_found]
                    logger.info(f"Maps: {len(place_names)} lugares encontrados — {place_names}")
                    comments = []
                    seen = set()

                    def _pull_place(place, want):
                        """Pide `want` reviews de un lugar y acumula las nuevas. Devuelve cuántas agregó."""
                        if want <= 0:
                            return 0
                        place_query = place.get("name", "") or place.get("formatted_address", prompt)
                        place_id = place.get("place_id")
                        if place_id:
                            query_arg = f"place_id:{place_id}"
                            logger.info(f"Maps APIFY: '{place_query}' (place_id: {place_id}), pido {want}...")
                        else:
                            query_arg = place_query
                            logger.info(f"Maps APIFY: '{place_query}', pido {want}...")

                        added = 0
                        try:
                            r = apify_scraper.invoke({"query": query_arg, "platform": platform, "max_items": want})
                        except Exception as e:
                            logger.warning(f"Maps APIFY '{place_query}': {e}")
                            return 0
                        if not r.get("success"):
                            return 0

                        coords = place.get("geometry", {}).get("location", {})
                        lat = coords.get("lat") if coords else None
                        lng = coords.get("lng") if coords else None
                        url_coords = f"&ll={lat},{lng}" if lat and lng else ""
                        for c in r.get("data", []):
                            txt = (c.get("comment") or "").strip()
                            if not txt or txt in seen:
                                continue
                            seen.add(txt)
                            c["latitude"] = lat
                            c["longitude"] = lng
                            if c.get("url"):
                                if "ll=" not in c["url"]:
                                    c["url"] += url_coords
                            else:
                                c["url"] = f"https://maps.google.com/?q={place_query}{url_coords}"
                            comments.append(c)
                            added += 1
                            if len(comments) >= max_comments:
                                break
                        return added

                    # Pasada 1: reparto equitativo entre lugares (preferimos varios sitios).
                    n_places = len(places_found)
                    per_place = max(1, -(-max_comments // n_places))  # techo de la división
                    for place in places_found:
                        if len(comments) >= max_comments:
                            break
                        _pull_place(place, min(per_place, max_comments - len(comments)))

                    # Pasada 2 (relleno): si falta, pedimos el déficit restante a cada lugar
                    # (los sitios con más reviews aportan las que aún no teníamos).
                    for place in places_found:
                        if len(comments) >= max_comments:
                            break
                        _pull_place(place, max_comments - len(comments))

                    comments = comments[:max_comments]
                    if len(comments) < max_comments:
                        logger.info(f"Maps: {len(comments)}/{max_comments} tras acumular en {n_places} lugares")

                # Normalizar social_media a "maps" para que el insert loop lo encuentre
                for c in comments:
                    c["social_media"] = "maps"

                all_comments.extend(comments)
                stats[platform] = len(comments)
                logger.info(f"  maps: {len(comments)} comentarios")

            # ── Resto de plataformas ────────────────────────────────────
            else:
                tool_func, tool_type = PLATFORM_CONFIG[platform]
                if tool_type == "apify":
                    result = tool_func.invoke({"query": prompt, "platform": platform, "max_items": max_comments})
                else:
                    result = tool_func.invoke({"query": prompt, "max_results": max_comments})

                if result.get("success"):
                    comments = result.get("data", [])
                    gemma_processed = gemma_processed or result.get("gemma_processed", False)
                    all_comments.extend(comments)
                    stats[platform] = len(comments)
                    logger.info(f"  {platform}: {len(comments)} comentarios")
                else:
                    errors.append(f"{platform}: {result.get('error', 'Error desconocido')}")
                    stats[platform] = 0

        except Exception as e:
            errors.append(f"{platform}: {str(e)}")
            stats[platform] = 0

    # Warning si no se alcanzó el max_comments en alguna plataforma
    for platform in social_medias:
        p = platform.lower().strip()
        got = stats.get(p, 0)
        if got > 0 and got < max_comments:
            errors.append(f"{p}: solo {got}/{max_comments} encontrados (límite de posts/lugares alcanzado)")

    analyzer_metrics = {}
    result = _analyze_store_embed(
        all_comments, stats, errors, prompt, social_medias,
        user_id, session_id, model, gemma_processed,
        user_email=user_email, metrics=analyzer_metrics
    )

    execution_time = time.time() - start_time
    if user_email:
        try:
            from db.ops import insert_request_performance
            from llm_manager import discover_embedding_model
            emb_model = discover_embedding_model() or "text-embedding-004"
            insert_request_performance(
                session_id=session_id,
                execution_time=execution_time,
                llm_model=model,
                classification_time=analyzer_metrics.get("classification_time", 0.0),
                embedding_model=emb_model,
                embedding_time=analyzer_metrics.get("embedding_time", 0.0),
                max_comments=max_comments,
                social_medias=social_medias,
            )
        except Exception as e:
            logger.error(f"Error al registrar request_performance en run_pipeline: {e}")

    return result


def _analyze_store_embed(
    all_comments: list[dict],
    stats: dict,
    errors: list[str],
    prompt: str,
    social_medias: list[str],
    user_id: str,
    session_id: str,
    model: str,
    gemma_processed: bool,
    user_email: str | None = None,
    metrics: dict | None = None,
) -> dict:
    """Cola compartida del pipeline: analizar → guardar en Supabase → embeddings.

    La usan tanto el flujo automático (`run_pipeline`) como el flujo manual con
    selección de posts (`run_pipeline_from_selection`).
    """
    if not all_comments:
        # Registrar petición vacía siempre (email anónimo si no hay sesión activa)
        try:
            from db.ops import insert_user_request
            from utils.cost_tracker import get_current_tracker
            tracker = get_current_tracker()
            cost = tracker.get_cost_usd() if tracker else 0.0
            email_to_use = user_email if user_email else "anonimo@chismesitogpt.com"
            insert_user_request(
                user_id=user_id,
                user_email=email_to_use,
                session_id=session_id,
                query=prompt,
                social_medias=social_medias,
                comments_count=0,
                model=model,
                cost=cost
            )
        except Exception as e:
            logger.error(f"Error al insertar peticion vacia: {e}")
        return {
            "dataframe": pd.DataFrame(),
            "stats": stats,
            "session_id": session_id,
            "errors": errors,
            "gemma_processed": gemma_processed,
        }

    # Consolidar
    df = pd.DataFrame(all_comments)
    logger.info(f"Total consolidado: {len(df)} comentarios")

    # Analizar (pasa el modelo para la generacion de categorias)
    df = analyze_comments(df, model=model, metrics=metrics)

    # Insertar petición del usuario (antes de insertar comentarios para FK)
    try:
        from db.ops import insert_user_request
        from utils.cost_tracker import get_current_tracker
        tracker = get_current_tracker()
        cost = tracker.get_cost_usd() if tracker else 0.0
        email_to_use = user_email if user_email else "anonimo@chismesitogpt.com"
        insert_user_request(
            user_id=user_id,
            user_email=email_to_use,
            session_id=session_id,
            query=prompt,
            social_medias=social_medias,
            comments_count=len(df),
            model=model,
            cost=cost
        )
    except Exception as e:
        logger.error(f"Error al registrar peticion de busqueda: {e}")

    # Guardar en Supabase (una plataforma a la vez para tener social_media correcto)
    for platform in social_medias:
        platform = platform.lower().strip()
        platform_df = df[df.get("social_media", "") == platform]
        if platform_df.empty:
            platform_df = df[df.get("social_media", "") == platform.replace("_", " ")]
        if platform_df.empty:
            continue

        try:
            count = insert_comments(platform_df, user_id, session_id, prompt, platform)
            logger.info(f"Guardados {count} en Supabase para {platform}")
        except Exception as e:
            msg = str(e)
            if "404" in msg or "not found" in msg.lower():
                msg += " | La tabla unified_comments no existe. Ejecuta supabase/schema.sql en el SQL Editor de Supabase."
            errors.append(f"Supabase {platform}: {msg}")

    # Insertar embeddings en pgvector (requiere IDs de Supabase)
    valid = df[df["embedding"].apply(lambda e: isinstance(e, list) and len(e) > 0)]
    if not valid.empty:
        try:
            from db.ops import get_session_comments
            stored = get_session_comments(session_id, user_id)
            if not stored.empty:
                stored["_key"] = stored["comment"].str[:50] + "|" + stored["social_media"]
                valid_copy = valid.copy()
                valid_copy["_key"] = valid_copy["comment"].str[:50] + "|" + valid_copy["social_media"]
                merged = valid_copy.merge(stored[["id", "_key"]], on="_key", how="inner")
                if not merged.empty:
                    pairs = list(zip(merged["id"].tolist(), merged["embedding"].tolist()))
                    batch_insert_embeddings(pairs)
                    logger.info(f"Embeddings insertados: {len(pairs)} en {SCHEMA}.unified_comments")
                else:
                    logger.warning("No se pudo hacer merge de IDs con embeddings")
            else:
                logger.warning("get_session_comments retorno vacio")
        except Exception as e:
            errors.append(f"Embeddings: {str(e)}")

    return {
        "dataframe": df,
        "stats": stats,
        "session_id": session_id,
        "errors": errors,
        "gemma_processed": gemma_processed,
    }


# ─── Flujo manual (dos fases): descubrir → seleccionar → extraer ─────────────

# Todas las plataformas soportan ya el flujo de selección manual (Fase 1 + 2).
DISCOVERY_SUPPORTED = {
    "youtube", "twitter", "x_twitter", "facebook", "instagram", "tiktok",
    "reddit", "playstore", "maps", "google_maps",
}


def _normalize_platform_key(p: str) -> str:
    """Normaliza el alias de plataforma a la clave canónica del dict de items."""
    if p == "x_twitter":
        return "twitter"
    if p == "google_maps":
        return "maps"
    return p


def discover(
    prompt: str,
    social_medias: list[str],
    maps_geo_toggle: bool = False,
    maps_location: str = "",
    maps_radius: int = 2000,
    max_items: int = 8,
) -> dict:
    """
    FASE 1: busca y devuelve los items candidatos por plataforma SIN extraer
    comentarios todavía. El usuario elegirá cuáles en la UI.

    Returns:
        {"items": {platform: [Item,...]}, "errors": [str], "unsupported": [str]}
    """
    items: dict[str, list[dict]] = {}
    errors: list[str] = []
    unsupported: list[str] = []

    for platform in social_medias:
        p = platform.lower().strip()
        key = _normalize_platform_key(p)
        try:
            if p == "youtube":
                items["youtube"] = discover_youtube_videos(prompt, max_items=max_items)
            elif p in ("twitter", "x_twitter"):
                items["twitter"] = apify_discover(prompt, "twitter", max_items=max_items)
            elif p in ("facebook", "instagram", "tiktok"):
                items[p] = apify_discover(prompt, p, max_items=max_items)
            elif p == "reddit":
                items["reddit"] = discover_reddit_posts(prompt, max_items=max_items)
            elif p == "playstore":
                items["playstore"] = discover_playstore_apps(prompt, max_items=max_items)
            elif p in ("maps", "google_maps"):
                items["maps"] = discover_maps_places(
                    prompt, max_items=max_items, use_geo=maps_geo_toggle,
                    location_text=maps_location, radius_meters=maps_radius)
            else:
                unsupported.append(p)
                continue
            if not items.get(key):
                errors.append(f"{p}: sin resultados para '{prompt}'")
        except Exception as e:
            errors.append(f"{p}: {str(e)}")

    return {"items": items, "errors": errors, "unsupported": unsupported}


def _fetch_maps_comments(place_ids: list[str], coords: dict, max_comments: int) -> list[dict]:
    """Extrae reviews de los place_ids elegidos vía APIFY, repartiendo el objetivo
    entre lugares y adjuntando lat/lng para el mapa. `coords` = {place_id:(lat,lng)}."""
    if not place_ids:
        return []
    comments: list[dict] = []
    seen: set[str] = set()
    per_place = max(1, -(-max_comments // len(place_ids)))  # techo de la división
    for pid in place_ids:
        if len(comments) >= max_comments:
            break
        want = min(per_place, max_comments - len(comments))
        try:
            r = apify_scraper.invoke({"query": f"place_id:{pid}", "platform": "maps", "max_items": want})
        except Exception as e:
            logger.warning(f"Maps APIFY place_id {pid}: {e}")
            continue
        lat, lng = coords.get(pid, (None, None))
        for c in r.get("data", []):
            txt = (c.get("comment") or "").strip()
            if not txt or txt in seen:
                continue
            seen.add(txt)
            c["latitude"] = lat
            c["longitude"] = lng
            c["social_media"] = "maps"
            comments.append(c)
            if len(comments) >= max_comments:
                break
    return comments


def run_pipeline_from_selection(
    prompt: str,
    selections: dict[str, list[str]],
    user_id: str,
    session_id: str | None = None,
    model: str = "gemini-2.0-flash-lite",
    max_comments: int = 10,
    discovery: dict[str, list[dict]] | None = None,
    user_email: str | None = None,
) -> dict:
    """
    FASE 2: extrae comentarios SOLO de los items seleccionados por el usuario,
    luego analiza, guarda y devuelve el mismo dict que `run_pipeline`.

    Args:
        selections: {platform: [id/url/appId/place_id, ...]} elegidos en la UI.
        discovery: {platform: [Item]} de la Fase 1 (para recuperar metadata como
            las coordenadas de Maps).
    """
    start_time = time.time()
    if session_id is None:
        session_id = str(uuid.uuid4())
    discovery = discovery or {}

    all_comments: list[dict] = []
    stats: dict[str, int] = {}
    errors: list[str] = []
    social_medias: list[str] = []

    for platform, ids in selections.items():
        p = platform.lower().strip()
        key = _normalize_platform_key(p)
        ids = [i for i in (ids or []) if i]
        if not ids:
            continue
        social_medias.append(key)
        logger.info(f"Extrayendo de {len(ids)} item(s) seleccionados en {key}...")
        try:
            if p == "youtube":
                comments = fetch_youtube_comments(ids, max_comments=max_comments)
            elif p in ("twitter", "x_twitter", "facebook", "instagram", "tiktok"):
                comments = apify_fetch_comments(key, ids, max_items=max_comments)
            elif p == "reddit":
                comments = fetch_reddit_comments_for(ids, max_comments=max_comments)
            elif p == "playstore":
                comments = fetch_playstore_reviews_for(ids, max_comments=max_comments)
            elif p in ("maps", "google_maps"):
                coords = {
                    it.get("id"): (it.get("lat"), it.get("lng"))
                    for it in discovery.get("maps", [])
                }
                comments = _fetch_maps_comments(ids, coords, max_comments)
            else:
                errors.append(f"{p}: selección manual no soportada aún")
                stats[key] = 0
                continue

            for c in comments:
                c.setdefault("social_media", key)
            all_comments.extend(comments)
            stats[key] = len(comments)
            logger.info(f"  {key}: {len(comments)} comentarios de la selección")
        except Exception as e:
            errors.append(f"{p}: {str(e)}")
            stats[key] = 0

    analyzer_metrics = {}
    result = _analyze_store_embed(
        all_comments, stats, errors, prompt, social_medias,
        user_id, session_id, model, gemma_processed=False,
        user_email=user_email, metrics=analyzer_metrics
    )

    execution_time = time.time() - start_time
    if user_email:
        try:
            from db.ops import insert_request_performance
            from llm_manager import discover_embedding_model
            emb_model = discover_embedding_model() or "text-embedding-004"
            insert_request_performance(
                session_id=session_id,
                execution_time=execution_time,
                llm_model=model,
                classification_time=analyzer_metrics.get("classification_time", 0.0),
                embedding_model=emb_model,
                embedding_time=analyzer_metrics.get("embedding_time", 0.0),
                max_comments=max_comments,
                social_medias=social_medias,
            )
        except Exception as e:
            logger.error(f"Error al registrar request_performance en run_pipeline_from_selection: {e}")

    return result
