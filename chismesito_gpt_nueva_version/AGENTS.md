# AGENTS.md — ChismesitoGPT v2 (Gradio + LangChain Agent)

> **Para:** DeepSeek, Gemini, Claude asistiendo en VibeCoding  
> **Objetivo:** Construir ChismesitoGPT v2 como app Gradio + LangChain Agent con APIFY + Supabase pgvector  
> **Carpeta:** `chismesito_gpt_nueva_version/`  
> **Entorno:** Conda `chismesito_gpt`

---

## Misión

App Gradio donde un **agente LLM orquesta tools** para:
1. Recibir prompt del usuario + selección de redes sociales (con iconos)
2. Buscar links reales con SerpAPI
3. Extraer comentarios según la plataforma (APIFY o API oficial)
4. Analizar (categorías LLM → zero-shot → sentimiento → emoción)
5. Guardar en Supabase PostgreSQL + vectorizar para RAG (pgvector, NO Pinecone)
6. Mostrar dashboard con gráficos y chat RAG
7. Descargar CSV/XLSX

---

## Decisiones YA TOMADAS (no cambiar)

| Decisión | Elección | Razón |
|----------|----------|-------|
| UI | **Gradio 4+** con `gr.Blocks` | HF-native, hot-reload, zero-shot deploy |
| Agente | **LangChain AgentExecutor** (tool-calling) | Gemini/DeepSeek-compatible |
| LLM | **Gemini 2.0 Flash** (primario), DeepSeek (fallback) | Gratis en HF, buena calidad |
| Search | **SerpAPI** ($50/mes) | Links reales de redes sociales |
| YouTube, Reddit, PlayStore | **APIs oficiales** envueltas en @tool | Probadas, estables |
| Facebook, X, TikTok, Instagram, Maps | **APIFY actors** (plan $29/mes) | Sin Selenium, sin ChromeDriver |
| DB relacional | **Supabase PostgreSQL** | Managed, REST API, auth incluido |
| Vector DB | **pgvector en Supabase** (NO Pinecone) | Todo en uno, sin servicio extra |
| NLP | **PySentimiento** (sentimiento + emociones) | Rápido, español nativo |
| Clasificación | **BART-large-mnli** zero-shot | Clasifica en categorías generadas por LLM |
| Embeddings | **Gemini `models/embedding-001`** (768d) | Gratis, multilingüe |
| Export | **pandas CSV + openpyxl Excel** | Simple y funcional |

---

## Estructura de Archivos

```
chismesito_gpt_nueva_version/
├── app.py                    # Entry point Gradio
├── agent.py                  # Agente LangChain (system prompt + tools + executor)
├── config.py                 # Carga .env, todas las API keys centralizadas
├── requirements.txt
├── .env.example
├── .gitignore
├── supabase/
│   ├── schema.sql            # unified_comments + chat_history + match_comments RPC
│   └── rls_policies.sql
├── tools/
│   ├── __init__.py
│   ├── search_tool.py        # SerpAPI → buscar links de redes sociales
│   ├── youtube_tool.py       # YouTube Data API v3
│   ├── reddit_tool.py        # PRAW
│   ├── playstore_tool.py     # google-play-scraper
│   ├── apify_tool.py         # APIFY (FB, X, TikTok, IG, Maps)
│   ├── sentiment_tool.py     # PySentimiento sentimiento
│   ├── emotion_tool.py       # PySentimiento emociones
│   ├── categories_tool.py    # LLM genera 5-10 categorías (muestra 20%)
│   ├── zero_shot_tool.py     # BART zero-shot → clasifica cada comentario
│   ├── embeddings_tool.py    # Gemini embeddings (768d)
│   └── export_tool.py        # CSV + Excel download
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py       # run_pipeline(prompt, social_medias, user_id)
│   ├── analyzer.py           # analyze_comments(df) → enriquecido
│   └── rag.py                # semantic_search + rag_chat
├── db/
│   ├── __init__.py
│   ├── supabase_client.py    # Singleton client
│   ├── ops.py                # CRUD en unified_comments
│   └── vector.py             # pgvector insert + match_comments RPC
├── ui/
│   ├── __init__.py
│   ├── components.py         # Bloques Gradio reutilizables
│   └── dashboard.py          # Gráficos Plotly/Matplotlib
└── utils/
    ├── __init__.py
    └── text_utils.py         # Traducción, detección idioma, helpers
```

---

## Orden de Implementación (SEGUIR ESTRICTAMENTE)

### FASE 0 — Setup (1 hora)

```powershell
# 1. Activar entorno
& "E:\Users\1167486\AppData\Local\anaconda3\Scripts\conda.exe" shell.powershell hook | Out-String | Invoke-Expression
conda activate chismesito_gpt

# 2. Instalar dependencias base
pip install gradio langchain langchain-google-genai langchain-community python-dotenv supabase pgvector psycopg2-binary pandas numpy plotly matplotlib seaborn openpyxl google-genai google-generativeai pysentimiento transformers torch beautifulsoup4 requests praw google-api-python-client googlemaps google-play-scraper tweepy deep-translator google-search-results aiohttp

# 3. Modelo spaCy español
python -m spacy download es_core_news_md
```

- [x] Crear `config.py` con carga de .env
- [x] Crear `.env.example` con todas las variables
- [x] Crear `.gitignore`
- [x] Crear `requirements.txt`

### FASE 1 — Supabase + DB (1 hora)

- [x] Crear proyecto en Supabase
- [x] Habilitar extensión pgvector: `CREATE EXTENSION IF NOT EXISTS vector;`
- [x] Ejecutar `supabase/schema.sql` → tabla `unified_comments` + `chat_history`
- [x] Crear función RPC `match_comments` (búsqueda semántica pgvector)
- [x] Configurar RLS policies (`supabase/rls_policies.sql`)
- [x] Implementar `db/supabase_client.py`
- [x] Implementar `db/ops.py` — insert, query, update
- [x] Implementar `db/vector.py` — insert_embedding + match_comments

### FASE 2 — Tools del Agente (3-4 horas)

Cada tool debe seguir este patrón:

```python
from langchain.tools import tool
from config import API_KEY

@tool
def nombre_tool(param: str) -> dict:
    """
    Descripción clara para que el agente decida cuándo usarla.
    
    Args:
        param: Descripción del parámetro
    
    Returns:
        {"success": bool, "data": list[dict], "count": int, "error": str}
    """
    try:
        # lógica
        return {"success": True, "data": [...], "count": len(...), "error": None}
    except Exception as e:
        return {"success": False, "data": [], "count": 0, "error": str(e)}
```

Orden de implementación:

- [x] `tools/search_tool.py` — SerpAPI: busca links de redes sociales
- [x] `tools/youtube_tool.py` — YouTube Data API v3
- [x] `tools/reddit_tool.py` — PRAW
- [x] `tools/playstore_tool.py` — google-play-scraper
- [x] `tools/apify_tool.py` — APIFY actors (FB, X, TikTok, IG, Maps)
- [x] `tools/sentiment_tool.py` — PySentimiento (lazy load)
- [x] `tools/emotion_tool.py` — PySentimiento emotions (lazy load)
- [x] `tools/categories_tool.py` — LLM genera 5-10 categorías (muestra 20%)
- [x] `tools/zero_shot_tool.py` — BART-large-mnli classifier (lazy load)
- [x] `tools/embeddings_tool.py` — Gemini embeddings (768d)
- [x] `tools/export_tool.py` — CSV + Excel

### FASE 3 — Pipeline (2 horas)

- [x] `pipeline/orchestrator.py`:
  - `run_pipeline(prompt: str, social_medias: list, user_id: str) -> dict`
  - Para cada red social: llamar a la tool correcta (APIFY vs API oficial)
  - Consolidar en un solo DataFrame
  - Llamar a `analyze_comments()`
  - Guardar en Supabase + pgvector
  - Retornar DataFrame + métricas + gráficos

- [x] `pipeline/analyzer.py`:
  - `analyze_comments(df: pd.DataFrame) -> pd.DataFrame`
  - Paso 1: muestra 20% → LLM genera categorías
  - Paso 2: zero-shot clasifica todas las filas
  - Paso 3: sentimiento (PySentimiento)
  - Paso 4: emoción (PySentimiento)
  - Paso 5: embeddings (Gemini, 768d)
  - Columnas añadidas: category, sentiment, emotion, embedding

- [x] `pipeline/rag.py`:
  - `semantic_search(query, session_id, user_id)` → top_k comentarios
  - `rag_chat(question, session_id, user_id, chat_history)` → respuesta

### FASE 4 — Agente LangChain (2 horas)

- [x] `agent.py`:
  ```python
  from langchain.agents import AgentExecutor, create_tool_calling_agent
  from langchain_google_genai import ChatGoogleGenerativeAI
  from langchain_core.prompts import ChatPromptTemplate
  
  SYSTEM_PROMPT = """Eres ChismesitoGPT, asistente de análisis de redes sociales.
  
  Flujo:
  1. Usa search_web para encontrar links reales de las redes seleccionadas
  2. Extrae comentarios con la tool adecuada:
     - YouTube → get_youtube_comments
     - Reddit → get_reddit_comments
     - Play Store → get_playstore_reviews
     - FB, X, TikTok, Instagram, Maps → apify_scraper
  3. Consolida comentarios
  4. Genera categorías (muestra 20%)
  5. Clasifica con zero-shot
  6. Analiza sentimiento y emoción
  7. Vectoriza y guarda en Supabase
  8. Genera dashboard
  
  Reglas:
  - NUNCA inventes URLs
  - Si una red falla, continúa con las demás
  - Muestra progreso: "Buscando en YouTube...", etc.
  """
  ```

- [x] Inicializar LLM (Gemini 2.0 Flash con fallback DeepSeek)
- [x] Cargar todas las tools
- [x] Crear AgentExecutor
- [x] Exponer `run_agent(prompt, social_medias, user_id)`

### FASE 5 — UI Gradio (3-4 horas)

- [x] `ui/components.py` — Componentes reutilizables:
  
  ```python
  def social_media_selector():
      """Checkbox group con iconos de redes sociales"""
      return gr.CheckboxGroup(
          choices=[
              ("🎬 YouTube", "youtube"),
              ("🐦 X / Twitter", "twitter"),
              ("👽 Reddit", "reddit"),
              ("👤 Facebook", "facebook"),
              ("📷 Instagram", "instagram"),
              ("🎵 TikTok", "tiktok"),
              ("📍 Google Maps", "maps"),
              ("🛒 Play Store", "playstore"),
          ],
          label="Selecciona redes sociales",
          value=["youtube"]
      )
  ```

- [x] `ui/dashboard.py` — Gráficos:
  - `plot_sentiment(df)` — barras apiladas por plataforma
  - `plot_emotions(df)` — barras agrupadas
  - `plot_categories(df)` — top 10 horizontal
  - `plot_map(df)` — mapa con coordenadas (Google Maps)

- [x] `app.py` — Entry point con bloques progresivos:
  ```python
  with gr.Blocks(theme=gr.themes.Soft(), title="ChismesitoGPT v2") as demo:
      gr.Markdown("# 🕵️ ChismesitoGPT v2")
      
      with gr.Row():
          prompt = gr.Textbox(label="¿Qué quieres investigar?", 
                             placeholder="Ej: Opiniones del iPhone 16 en México")
      
      with gr.Row():
          social_medias = gr.CheckboxGroup(
              choices=[("🎬 YouTube", "youtube"), ("🐦 X/Twitter", "twitter"),
                       ("👽 Reddit", "reddit"), ("👤 Facebook", "facebook"),
                       ("📷 Instagram", "instagram"), ("🎵 TikTok", "tiktok"),
                       ("📍 Google Maps", "maps"), ("🛒 Play Store", "playstore")],
              label="Redes sociales"
          )
          search_btn = gr.Button("🔍 Buscar y Analizar", variant="primary")
      
      # Secciones progresivas (visible=False inicialmente)
      with gr.Column(visible=False) as results_section:
          with gr.Tab("📋 Datos"):
              data_table = gr.Dataframe(label="Comentarios")
              download_csv = gr.File(label="⬇ CSV")
              download_xlsx = gr.File(label="📥 Excel")
          
          with gr.Tab("📊 Dashboard"):
              sentiment_plot = gr.Plot(label="Sentimiento")
              emotion_plot = gr.Plot(label="Emociones")
              category_plot = gr.Plot(label="Categorías")
          
          with gr.Tab("💬 Chat RAG"):
              chatbot = gr.Chatbot(label="Pregunta sobre los datos")
              chat_input = gr.Textbox(label="Tu pregunta")
              chat_btn = gr.Button("Preguntar")
      
      # Eventos
      search_btn.click(fn=run_pipeline_ui, inputs=[prompt, social_medias],
                       outputs=[data_table, sentiment_plot, emotion_plot, 
                               category_plot, download_csv, download_xlsx, results_section])
  ```

### FASE 6 — Deploy HuggingFace (1 hora)

- [x] Crear `Dockerfile`
- [x] Configurar `requirements.txt`
- [x] Configurar `packages.txt` (si necesita chromium)
- [x] Test local: `python app.py`
- [x] Push a HuggingFace Space

---

## APIFY — Plan $29/mes

Actors a usar (ya pre-configurados en APIFY):

| Red social | Actor ID | Propósito |
|-----------|----------|-----------|
| Facebook | `apify/facebook-comments-scraper` | Comentarios de posts públicos |
| X / Twitter | `apify/twitter-scraper` | Tweets por búsqueda |
| Instagram | `apify/instagram-comment-scraper` | Comentarios de posts |
| TikTok | `apify/tiktok-comments-scraper` | Comentarios de videos |
| Google Maps | `apify/google-maps-scraper` | Reviews de lugares |

### Patrón de tool APIFY:

```python
# tools/apify_tool.py
from langchain.tools import tool
from config import APIFY_API_KEY
import requests

ACTOR_MAP = {
    "facebook": "apify/facebook-comments-scraper",
    "twitter": "apify/twitter-scraper",
    "instagram": "apify/instagram-comment-scraper",
    "tiktok": "apify/tiktok-comments-scraper",
    "maps": "apify/google-maps-scraper",
}

@tool
def apify_scraper(query: str, platform: str, max_items: int = 50) -> dict:
    """
    Extrae comentarios de {platform} usando APIFY.
    
    Args:
        query: Término de búsqueda o URL
        platform: 'facebook', 'twitter', 'instagram', 'tiktok', 'maps'
        max_items: Máximo de items a obtener
    
    Returns:
        {"success": bool, "data": list[dict], "count": int, "error": str}
    """
    if platform not in ACTOR_MAP:
        return {"success": False, "data": [], "count": 0, "error": f"Plataforma no soportada: {platform}"}
    
    try:
        actor_id = ACTOR_MAP[platform]
        # Iniciar actor run en APIFY
        response = requests.post(
            f"https://api.apify.com/v2/acts/{actor_id}/runs",
            headers={"Authorization": f"Bearer {APIFY_API_KEY}"},
            json={"searchTerms": [query], "maxItems": max_items}
        )
        response.raise_for_status()
        run_data = response.json()
        run_id = run_data["data"]["id"]
        
        # Esperar y obtener resultados (simplificado)
        import time
        time.sleep(10)  # En producción usar polling
        
        results_response = requests.get(
            f"https://api.apify.com/v2/acts/{actor_id}/runs/{run_id}/dataset/items",
            headers={"Authorization": f"Bearer {APIFY_API_KEY}"}
        )
        items = results_response.json()
        
        comments = [{"comment": item.get("text", item.get("comment", "")), 
                     "username": item.get("username", ""),
                     "platform": platform} 
                    for item in items]
        
        return {"success": True, "data": comments, "count": len(comments), "error": None}
    except Exception as e:
        return {"success": False, "data": [], "count": 0, "error": str(e)}
```

---

## Routing lógico (Agent decide)

```
Usuario: "iPhone 16 opiniones" + selecciona [YouTube, Twitter, Reddit]
                         │
                         ▼
┌─────────────────────────────────────────┐
│            Agente LangChain              │
│  "Necesito buscar en YouTube, Twitter,   │
│   y Reddit. Primero busco links..."      │
└────────┬──────────┬──────────┬──────────┘
         │          │          │
         ▼          ▼          ▼
    YouTube     Twitter     Reddit
    API v3      APIFY       PRAW
         │          │          │
         ▼          ▼          ▼
┌─────────────────────────────────────────┐
│         Consolidar DataFrame            │
│  [comment, social_media, username, ...]  │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│         Pipeline de Análisis             │
│  1. Categorías (LLM)                    │
│  2. Zero-shot classification (BART)     │
│  3. Sentimiento (PySentimiento)          │
│  4. Emoción (PySentimiento)              │
│  5. Embeddings (Gemini → pgvector)      │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│         Supabase                         │
│  unified_comments + pgvector embeddings  │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│         UI: Dashboard + Chat RAG         │
│  Gráficos Plotly + Descarga CSV/Excel   │
└─────────────────────────────────────────┘
```

---

## Convenciones de Código

### Lazy Loading (IMPORTANTE para cold start en HF):

```python
_model = None

def _get_model():
    global _model
    if _model is None:
        from transformers import pipeline
        _model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _model
```

### Tool con @tool decorator:

```python
from langchain.tools import tool

@tool
def mi_tool(param: str) -> dict:
    """Descripción clara para el agente."""
    try:
        return {"success": True, "data": [...], "count": N, "error": None}
    except Exception as e:
        return {"success": False, "data": [], "count": 0, "error": str(e)}
```

### Logging, NO print():

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Procesando...")
```

---

## Lo que NO debes hacer

- ❌ NO uses Selenium/ChromeDriver → APIFY para FB, IG, TikTok, X, Maps
- ❌ NO uses Pinecone → pgvector en Supabase
- ❌ NO crees múltiples tablas → solo `unified_comments`
- ❌ NO hagas scraping directo → APIs oficiales o APIFY
- ❌ NO hardcodees API keys → `.env` vía `config.py`
- ❌ NO uses `print()` → `logging`
- ❌ NO bloquees la UI → usa generadores `yield` en Gradio
- ❌ NO uses la API antigua de Gemini → `google.genai` (nueva)
