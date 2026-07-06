# APIFY Processes — ChismesitoGPT v2

Cada red social que usa APIFY tiene un flujo en 2 pasos o un flujo especial. Aquí se documentan.

---

## Flujos estándar (2 pasos)

### 1. Facebook (3 pasos)

```
Usuario: "Banco Azteca quejas"
              │
              ▼
┌─────────────────────────────────────────────┐
│ PASO 1: Buscar posts                        │
│ Actor: apify/facebook-search-scraper        │
│ Input: searchTerms = ["Banco Azteca quejas"]│
│ Output: URLs de posts encontrados           │
│ Tiempo: ~30-90s                             │
└────────────────────┬────────────────────────┘
                     │  URLs (máx 3)
                     ▼
┌─────────────────────────────────────────────┐
│ PASO 2: Scrapear detalles de cada post      │
│ Actor: apify/facebook-posts-scraper         │
│ Input: startUrls = [url]                    │
│ Output: contenido del post, metadata        │
│ Tiempo: ~20-40s por post                    │
└────────────────────┬────────────────────────┘
                     │  post procesado
                     ▼
┌─────────────────────────────────────────────┐
│ PASO 3: Extraer comentarios del post        │
│ Actor: apify/facebook-comments-scraper      │
│ Input: startUrls = [url]                    │
│ Output: comentarios, username, fecha, likes │
│ Tiempo: ~30-60s por post                    │
└────────────────────┬────────────────────────┘
                     │  comentarios[]  (≤ max_comments)
                     ▼
              Análisis + Supabase
```

**Requisitos en APIFY Console:**

- Activar 3 actors: `facebook-search-scraper`, `facebook-posts-scraper`, `facebook-comments-scraper`

---

### 2. X / Twitter

```
Usuario: "iPhone 16 opiniones"
              │
              ▼
┌─────────────────────────────────────────────┐
│ PASO 1: Buscar tweets por keyword           │
│ Actor: watcher.data/search-x-by-keywords    │
│ Input: searchTerms = ["iPhone 16 opiniones"]│
│ Output: tweets con URLs, texto, autor       │
│ Tiempo: ~20-60s                             │
└────────────────────┬────────────────────────┘
                     │  tweets encontrados (máx 3)
                     ▼
┌─────────────────────────────────────────────┐
│ PASO 2: Extraer replies de esos tweets      │
│ Actor: scraper_one/x-post-replies-scraper   │
│ Input: startUrls = [tweet_url1, ...]        │
│ Output: replies (username, texto, fecha)    │
│ Tiempo: ~20-40s por tweet                   │
└────────────────────┬────────────────────────┘
                     │  comentarios[]  (≤ max_comments)
                     ▼
              Análisis + Supabase
```

**Requisitos en APIFY Console:**

- Activar ambos actors: `search-x-by-keywords` y `x-post-replies-scraper`

---

### 3. Instagram

```
Usuario: "Nike México"
              │
              ▼
┌─────────────────────────────────────────────┐
│ PASO 1: Buscar hashtags                     │
│ Actor: apify/instagram-hashtag-scraper      │
│ Input: searchTerms = ["Nike México"]        │
│ Output: posts con URL, caption, likes       │
│ Tiempo: ~30-90s                             │
└────────────────────┬────────────────────────┘
                     │  posts encontrados (máx 3)
                     ▼
┌─────────────────────────────────────────────┐
│ PASO 2: Scrapear datos del post             │
│ Actor: apify/instagram-post-scraper         │
│ Input: startUrls = [post_url1, ...]         │
│ Output: comentarios, username, likes        │
│ Tiempo: ~30-60s por post                    │
└────────────────────┬────────────────────────┘
                     │  comentarios[]  (≤ max_comments)
                     ▼
              Análisis + Supabase
```

**Requisitos en APIFY Console:**

- Activar ambos actors: `instagram-hashtag-scraper` y `instagram-post-scraper`

---

### 4. TikTok

```
Usuario: "iPhone 16 review"
              │
              ▼
┌─────────────────────────────────────────────┐
│ PASO 1: Buscar hashtags                     │
│ Actor: clockworks/tiktok-hashtag-scraper    │
│ Input: searchTerms = ["iPhone 16 review"]   │
│ Output: videos con URL, descripción, likes  │
│ Tiempo: ~30-90s                             │
└────────────────────┬────────────────────────┘
                     │  videos encontrados (máx 3)
                     ▼
┌─────────────────────────────────────────────┐
│ PASO 2: Extraer comentarios de videos       │
│ Actor: clockworks/tiktok-comments-scraper   │
│ Input: startUrls = [video_url1, ...]        │
│ Output: comentarios, username, likes        │
│ Tiempo: ~20-60s por video                   │
└────────────────────┬────────────────────────┘
                     │  comentarios[]  (≤ max_comments)
                     ▼
              Análisis + Supabase
```

**Requisitos en APIFY Console:**

- Activar ambos actors: `tiktok-hashtag-scraper` y `tiktok-comments-scraper`

---

## Flujo especial (API + APIFY)

### 5. Google Maps

```
Usuario: "Restaurantes CDMX"
              │
              ▼
┌─────────────────────────────────────────────┐
│ PASO 1: Buscar lugares (Google Maps API)    │
│ API: googlemaps.Client(places)              │
│ Input: query = "Restaurantes CDMX"          │
│ Output: [Place1, Place2, ...] (máx 5)      │
│ Tiempo: <1s                                 │
│ API Key: MAPS_API_KEY en .env               │
└────────────────────┬────────────────────────┘
                     │  lugares[] (nombres)
                     ▼
┌─────────────────────────────────────────────┐
│ PASO 2: Extraer reviews de cada lugar       │
│ Actor: compass/Google-Maps-Reviews-Scraper  │
│ Input: searchStrings = [lugar1, lugar2, ...]│
│        (uno por lugar, en paralelo)         │
│ Output: reviews, rating, username, fecha    │
│ Tiempo: ~30-60s por lugar                   │
│ Total reviews ≤ slider max_comments         │
└────────────────────┬────────────────────────┘
                     │  comentarios[]  (≤ max_comments)
                     ▼
              Análisis + Supabase
```

**Requisitos:**

- `MAPS_API_KEY` en `.env` (Google Cloud Console)
- Activar actor: `compass/Google-Maps-Reviews-Scraper` en APIFY Console

**Si Places API no encuentra lugares**: usa APIFY directo con el query del usuario como fallback.

---

## Control de límites

| Control | Dónde | Valor |
|---------|-------|-------|
| Slider UI | `app.py` | Default 10, máx 50 |
| Posts/lugares a procesar | `apify_tool.py` / `orchestrator.py` | Máx 3 (FB, IG, TT, X) o 5 (Maps) |
| Total comentarios por red | `orchestrator.py` | `max_comments` del slider |
| Timeout APIFY por actor | `apify_tool.py` | 180s (search), 120s (comments) |

---

## Resumen de Actors

| Red social | Actor Búsqueda | Actor Post | Actor Comentarios |
|-----------|---------------|-----------|-------------------|
| Facebook | `apify/facebook-search-scraper` | `apify/facebook-posts-scraper` | `apify/facebook-comments-scraper` |
| X/Twitter | `watcher.data/search-x-by-keywords` | — | `scraper_one/x-post-replies-scraper` |
| Instagram | `apify/instagram-hashtag-scraper` | — | `apify/instagram-post-scraper` |
| TikTok | `clockworks/tiktok-hashtag-scraper` | — | `clockworks/tiktok-comments-scraper` |
| Google Maps | Google Maps Places API | — | `compass/Google-Maps-Reviews-Scraper` |

**YouTube, Reddit y Play Store NO usan APIFY — usan APIs oficiales.**
