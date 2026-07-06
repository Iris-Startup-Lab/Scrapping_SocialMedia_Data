# -*- coding: utf-8 -*-
"""tools/google_maps_tool.py — Google Maps Places API (oficial)."""

import logging
from langchain.tools import tool
from config import MAPS_API_KEY

logger = logging.getLogger(__name__)

try:
    import googlemaps
except ImportError:
    googlemaps = None


def _resolve_coordinates(gmaps, location_text: str):
    """Resuelve dirección o texto a tupla (lat, lng)."""
    if not location_text:
        return 19.4326, -99.1332  # Centro CDMX por default
    
    # Si ya son coordenadas "lat,lng"
    try:
        parts = location_text.split(",")
        if len(parts) == 2:
            return float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        pass
    
    # Geocodificar texto
    try:
        geom = gmaps.geocode(location_text)
        if geom:
            loc = geom[0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
    except Exception as e:
        logger.error(f"Geocoding error para '{location_text}': {e}")
    return None


def geocode_location(location_text: str) -> dict:
    """
    Geocodifica una dirección o coordenadas lat,lng usando la API oficial.

    Args:
        location_text: Dirección o coordenadas lat,lng para el centro de búsqueda

    Returns:
        dict con {"success": bool, "lat": float, "lng": float, "address": str, "error": str}
    """
    if not MAPS_API_KEY:
        return {"success": False, "lat": None, "lng": None, "address": "", "error": "MAPS_API_KEY no configurada"}
    if googlemaps is None:
        return {"success": False, "lat": None, "lng": None, "address": "", "error": "googlemaps no instalado"}

    # Si ya son coordenadas "lat,lng"
    try:
        parts = location_text.split(",")
        if len(parts) == 2:
            lat = float(parts[0].strip())
            lng = float(parts[1].strip())
            return {"success": True, "lat": lat, "lng": lng, "address": f"Coordenadas manuales ({lat}, {lng})", "error": None}
    except ValueError:
        pass

    try:
        gmaps = googlemaps.Client(key=MAPS_API_KEY)
        geom = gmaps.geocode(location_text)
        if geom:
            loc = geom[0]["geometry"]["location"]
            address = geom[0].get("formatted_address", location_text)
            return {
                "success": True,
                "lat": loc["lat"],
                "lng": loc["lng"],
                "address": address,
                "error": None
            }
        else:
            return {"success": False, "lat": None, "lng": None, "address": "", "error": f"No se pudo resolver la ubicación: {location_text}"}
    except Exception as e:
        logger.error(f"Geocoding error para '{location_text}': {e}")
        return {"success": False, "lat": None, "lng": None, "address": "", "error": str(e)}


# ─── FASE 1: Descubrir lugares ───────────────────────────────────────────────

def discover_maps_places(
    query: str,
    max_items: int = 10,
    use_geo: bool = False,
    location_text: str = "",
    radius_meters: int = 2000,
) -> list[dict]:
    """
    Busca lugares en Google Maps (texto o por cercanía) y devuelve Items
    normalizados con coordenadas para el mapa + checklist (Fase 1).

    Returns:
        list[Item] con {platform:'maps', id (place_id), title, author (dirección),
        thumbnail, stat (rating★), url, lat, lng, number}
    """
    if not MAPS_API_KEY or googlemaps is None:
        logger.warning("Maps: sin API key / librería; discover vacío")
        return []

    try:
        gmaps = googlemaps.Client(key=MAPS_API_KEY)
        if use_geo:
            center = _resolve_coordinates(gmaps, location_text)
            if not center:
                return []
            resp = gmaps.places_nearby(location=center, radius=radius_meters, keyword=query)
        else:
            resp = gmaps.places(query=query)
        results = resp.get("results", [])[:max_items]
    except Exception as e:
        logger.error(f"Maps discover error: {e}")
        return []

    places = []
    for i, p in enumerate(results, start=1):
        place_id = p.get("place_id")
        if not place_id:
            continue
        loc = p.get("geometry", {}).get("location", {})
        lat, lng = loc.get("lat"), loc.get("lng")
        rating = p.get("rating")
        n = p.get("user_ratings_total")
        stat = ""
        if isinstance(rating, (int, float)):
            stat = f"{rating:.1f} ★" + (f" · {int(n):,} reseñas" if n else "")
        address = p.get("formatted_address") or p.get("vicinity") or ""
        name = p.get("name", "(sin nombre)")
        photo_ref = None
        photos = p.get("photos") or []
        if photos and isinstance(photos, list):
            photo_ref = photos[0].get("photo_reference")
        thumb = (
            f"https://maps.googleapis.com/maps/api/place/photo"
            f"?maxwidth=200&photo_reference={photo_ref}&key={MAPS_API_KEY}"
            if photo_ref else ""
        )
        ll = f"&ll={lat},{lng}" if lat and lng else ""
        places.append({
            "platform": "maps",
            "id": place_id,
            "title": f"{i}. {name}",
            "author": address,
            "thumbnail": thumb,
            "stat": stat,
            "url": f"https://maps.google.com/?q={name}{ll}",
            "lat": lat,
            "lng": lng,
            "number": i,
        })
    logger.info(f"Maps discover: {len(places)} lugares para '{query}'")
    return places


@tool
def get_google_maps_reviews(
    query: str,
    max_results: int = 10,
    use_geo: bool = False,
    location_text: str = "",
    radius_meters: int = 2000
) -> dict:
    """
    Obtiene reviews de Google Maps usando la API oficial.
    Soporta búsqueda cercana si use_geo es True.

    Args:
        query: Termino de busqueda (ej: 'Restaurante La Polar CDMX')
        max_results: Maximo de reviews (default 10)
        use_geo: Activa la búsqueda geográfica por proximidad
        location_text: Dirección o coordenadas lat,lng para el centro de búsqueda
        radius_meters: Radio de búsqueda en metros (default 2000)

    Returns:
        {"success": bool, "data": [{"comment": str, "username": str, "rating": float, ...}],
         "count": int, "coords": dict|None, "error": str}
    """
    if not MAPS_API_KEY:
        return {"success": False, "data": [], "count": 0, "coords": None,
                "error": "MAPS_API_KEY no configurada"}
    if googlemaps is None:
        return {"success": False, "data": [], "count": 0, "coords": None,
                "error": "googlemaps no instalado (pip install googlemaps)"}

    try:
        gmaps = googlemaps.Client(key=MAPS_API_KEY)

        # Buscar lugar
        if use_geo:
            center = _resolve_coordinates(gmaps, location_text)
            if not center:
                return {"success": False, "data": [], "count": 0, "coords": None,
                        "error": f"No se pudo resolver la ubicación: {location_text}"}
            places = gmaps.places_nearby(location=center, radius=radius_meters, keyword=query)
        else:
            places = gmaps.places(query=query)

        results = places.get("results", [])
        if not results:
            return {"success": True, "data": [], "count": 0, "coords": None, "error": None}

        # Tomar el primer resultado y obtener detalles con reviews
        place_id = results[0]["place_id"]
        details = gmaps.place(place_id=place_id, fields=["name", "review", "geometry"])
        place = details.get("result", {})

        reviews = place.get("reviews", [])[:max_results]
        coords = place.get("geometry", {}).get("location")

        comments = []
        lat = coords.get("lat") if coords else None
        lng = coords.get("lng") if coords else None
        url_coords = f"&ll={lat},{lng}" if lat and lng else ""

        for r in reviews:
            comments.append({
                "comment": r.get("text", ""),
                "username": r.get("author_name", ""),
                "rating": r.get("rating", 0),
                "post_date": r.get("time", ""),
                "url": f"https://maps.google.com/?q={place.get('name', query)}{url_coords}",
                "social_media": "google_maps",
                "latitude": lat,
                "longitude": lng,
            })

        logger.info(f"Maps API: {len(comments)} reviews de '{place.get('name', query)}'")
        return {"success": True, "data": comments, "count": len(comments),
                "coords": coords, "error": None}

    except Exception as e:
        logger.error(f"Maps API error: {e}")
        return {"success": False, "data": [], "count": 0, "coords": None, "error": str(e)}
