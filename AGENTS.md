# AGENTS.md — ChismesitoGPT v1.0

> **Para:** DeepSeek, Claude, Gemini o cualquier LLM asistiendo en VibeCoding  
> **Objetivo:** Guiar la migración de Shiny → Gradio + LangChain con arquitectura agent-based  

---

## Misión

Reconstruir **ChismesitoGPT** como una app Gradio + LangChain donde un **agente LLM orquesta tools** para descargar comentarios de redes sociales, analizarlos (sentimiento, emociones, categorías), guardarlos en Supabase con pgvector, y ofrecer un dashboard + chat RAG. La app debe desplegarse en HuggingFace Spaces.

---

## Decisiones YA TOMADAS (no cambiar)

| Decisión | Elección | Motivo |
|----------|----------|--------|
| UI Framework | **Gradio 4+** con `gr.Blocks` | HuggingFace-native, UI progresiva, rápida de prototipar |
| Agente | **LangChain Agent + Tools** (tool-calling agent) | Estándar, bien documentado, Gemini/DeepSeek-compatible |
| LLM Principal | **Gemini 2.0 Flash** (gratis en HF) con **DeepSeek** como fallback | Costo $0 en HF, buena calidad |
| Search | **SerpAPI** (primario) + **Google Custom Search** (secundario) | Encontrar links reales de redes sociales |
| YouTube, Reddit, Maps | **APIs oficiales** envueltas en LangChain Tools | Probadas, estables, sin ban |
| FB, IG, TikTok, X | **APIFY actors** (no Selenium) | Mantenible, sin ChromeDriver, managed |
| DB | **Supabase** (PostgreSQL + pgvector + Auth) | Todo en uno: relacional, vectorial, auth |
| Esquema BD | **1 tabla unificada** `unified_comments` | Fácil cross-platform, queries simples |
| NLP Ligero | **PySentimiento** (sentimiento + emociones) | Rápido, español nativo, sin GPU |
| Zero-Shot | **BART-large-mnli** (HuggingFace pipeline) | Clasificar en categorías generadas por LLM |
| Embeddings | **Gemini `models/embedding-001`** (768d) | Gratis, buena calidad multilingüe |
| Auth | **Supabase Auth nativo** (email/password) | Robusto, RLS, OAuth-ready |
| Export | **openpyxl** (Excel) + pandas (CSV) | Simple y funcional |

---

## Estructura de Archivos Objetivo

```
chismesitogpt_v1/
├── app.py                    # Entry point Gradio (único archivo que se ejecuta)
├── agent.py                  # Agente LangChain: definición, tools, sistema prompt
├── config.py                 # Carga de .env, configuración centralizada
├── requirements.txt          # Dependencias Python
├── packages.txt              # Dependencias de sistema (si HF lo requiere)
├── Dockerfile                # Para HuggingFace Spaces
├── .env.example              # Template de variables de entorno
├── supabase/
│   ├── schema.sql            # SQL para crear tablas (unified_comments, chat_history)
│   ├── migrations/           # Migraciones versionadas
│   └── rls_policies.sql      # Row Level Security policies
├── tools/
│   ├── __init__.py
│   ├── search.py             # SerpAPI + Google Custom Search Tool
│   ├── youtube.py            # YouTube Data API v3 Tool
│   ├── reddit.py             # PRAW Tool
│   ├── google_maps.py        # Google Maps Places API Tool
│   ├── apify.py              # APIFY scrapers (FB, IG, TikTok, X)
│   ├── playstore.py          # Google Play Store scraper Tool
│   ├── wikipedia.py          # Wikipedia Tool
│   ├── web_scraper.py        # Requests + BS4 fallback Tool
│   ├── sentiment.py          # PySentimiento + spaCy Tool
│   ├── emotion.py            # PySentimiento emotions Tool
│   ├── categories.py         # LLM genera 5-10 categorías
│   ├── zero_shot.py          # BART zero-shot classifier Tool
│   ├── embeddings.py         # Gemini embeddings Tool
│   └── export.py             # CSV + Excel export Tools
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py       # Flujo principal: scrape → analyze → store → visualize
│   ├── analyzer.py           # Pipeline de análisis (categorías, sentiment, emotion)
│   └── rag.py                # Búsqueda semántica + chat RAG
├── db/
│   ├── __init__.py
│   ├── supabase_client.py    # Cliente Supabase (singleton)
│   ├── ops.py                # Operaciones CRUD
│   └── vector.py             # Operaciones pgvector (insert, search)
├── auth/
│   ├── __init__.py
│   └── supabase_auth.py      # Sign up, sign in, sign out, session management
├── ui/
│   ├── __init__.py
│   ├── components.py         # Componentes Gradio reutilizables
│   ├── dashboard.py          # Gráficos (Plotly/Matplotlib)
│   └── styles.py             # CSS y theming
└── utils/
    ├── __init__.py
    ├── text_utils.py         # Traducción, detección de idioma
    └── validators.py         # Validación de inputs
```

---

## Orden de Implementación (seguir ESTRICTAMENTE)

### Paso 1: Setup del proyecto
```
[x] Crear carpeta chismesitogpt_v1/
[x] Crear .env.example con TODAS las variables necesarias
[x] Crear requirements.txt (ver abajo)
[x] Crear config.py que cargue todas las env vars
[x] Crear .gitignore (incluir .env, __pycache__, .venv)
```

### Paso 2: Base de Datos (Supabase)
```
[x] Crear proyecto en Supabase
[x] Ejecutar supabase/schema.sql (tabla unified_comments)
[x] Habilitar extensión pgvector
[x] Crear función SQL match_comments (búsqueda semántica)
[x] Configurar Supabase Auth (habilitar email/password)
[x] Crear RLS policies
[x] Implementar db/supabase_client.py
[x] Implementar db/ops.py (insert, query, update)
[x] Implementar db/vector.py (insert_embedding, search_similar)
[x] Implementar auth/supabase_auth.py
```

### Paso 3: Tools del Agente (UNA por archivo)
```
Para cada tool, el patrón es:
1. Importar config (API keys)
2. Definir función con @tool decorator
3. Retornar dict estandarizado: {"success": bool, "data": list[dict], "error": str}
4. Manejar errores sin crashear

Orden de implementación:
[x] tools/search.py           (necesita SerpAPI)
[x] tools/youtube.py          (necesita YOUTUBE_API_KEY)
[x] tools/reddit.py           (necesita REDDIT_*)
[x] tools/google_maps.py      (necesita MAPS_API_KEY)
[x] tools/apify.py            (necesita APIFY_API_KEY)
[x] tools/playstore.py        (no necesita API key)
[x] tools/wikipedia.py        (no necesita API key)
[x] tools/web_scraper.py      (opcional: OXYLABS proxy)
[x] tools/sentiment.py        (usa PySentimiento, lazy load)
[x] tools/emotion.py          (usa PySentimiento, lazy load)
[x] tools/categories.py       (llama a Gemini, genera categorías)
[x] tools/zero_shot.py        (carga BART pipeline, lazy)
[x] tools/embeddings.py       (llama a Gemini embeddings)
[x] tools/export.py           (pandas → CSV, openpyxl → Excel)
```

### Paso 4: Pipeline de Análisis
```
[x] pipeline/analyzer.py
    - Función principal: analyze_comments(df: pd.DataFrame) -> pd.DataFrame
    - Paso 1: tomar muestra 20%, llamar a tools/categories.py
    - Paso 2: clasificar todas las filas con tools/zero_shot.py
    - Paso 3: analizar sentimiento (tools/sentiment.py)
    - Paso 4: analizar emoción (tools/emotion.py)
    - Paso 5: generar embeddings (tools/embeddings.py)
    - Retornar DataFrame enriquecido con columnas: category, sentiment, emotion, embedding

[x] pipeline/orchestrator.py
    - Función principal: run_pipeline(prompt, social_medias, user_id)
    - Para cada red social, llamar a la tool correspondiente
    - Consolidar todos los comentarios en un solo DataFrame
    - Llamar a analyze_comments()
    - Guardar en Supabase (db/ops.py)
    - Guardar embeddings en pgvector (db/vector.py)
    - Retornar DataFrame final + métricas

[x] pipeline/rag.py
    - semantic_search(query, session_id, user_id)
    - rag_chat(question, session_id, user_id, chat_history)
```

### Paso 5: Agente LangChain
```
[x] agent.py
    - Definir system prompt (ver sección abajo)
    - Inicializar LLM (Gemini 2.0 Flash con fallback a DeepSeek)
    - Cargar todas las tools
    - Crear AgentExecutor con tool-calling agent
    - Exponer función: run_agent(user_prompt, social_medias, user_id)
```

### Paso 6: UI Gradio
```
[x] ui/components.py
    - login_block(): formulario de login Supabase
    - search_block(): input de prompt + checkbox de redes con iconos
    - progress_block(): status en tiempo real (visible=False inicialmente)
    - results_table(): tabla de resultados con descarga
    - dashboard_block(): gráficos de sentimiento, emoción, categorías
    - chat_block(): chatbot RAG sobre los datos

[x] ui/dashboard.py
    - plot_sentiment(df): gráfico de barras/apilado
    - plot_emotions(df): gráfico de barras
    - plot_categories(df): gráfico de barras horizontal

[x] app.py (entry point)
    - Inicializar estado de sesión
    - Login flow
    - Search flow (progresivo)
    - Chat flow
```

### Paso 7: Deploy
```
[x] Crear Dockerfile
[x] Test local: python app.py
[x] Push a HuggingFace Space
[x] Configurar secrets en HF
```

---

## System Prompt del Agente (agent.py)

```python
SYSTEM_PROMPT = """Eres ChismesitoGPT, un agente especializado en extraer y analizar comentarios de redes sociales.

## Tu objetivo
Dado un prompt de búsqueda y una lista de redes sociales, debes:
1. Usar la tool de búsqueda web (SerpAPI) para encontrar URLs relevantes de las redes sociales solicitadas
2. Extraer comentarios de cada red social usando la tool específica
3. Consolidar todos los comentarios
4. Analizar el sentimiento y emoción de cada comentario
5. Generar categorías temáticas y clasificar los comentarios
6. Guardar todo en la base de datos

## Reglas
- NUNCA inventes URLs. Siempre usa search_web para encontrar links reales.
- Si una red social no está en la lista del usuario, NO la proceses.
- Para YouTube y Reddit, usa SIEMPRE las APIs oficiales.
- Para Facebook, Instagram, TikTok y X/Twitter, usa SIEMPRE APIFY.
- Muestra progreso al usuario: "Buscando en YouTube...", "Analizando sentimiento...", etc.
- Si una tool falla, informa al usuario pero continúa con las demás.
- Al finalizar, muestra un resumen de cuántos comentarios se obtuvieron por plataforma.

## Redes sociales y tools
- YouTube → get_youtube_comments
- Reddit → get_reddit_comments  
- Google Maps → get_google_maps_reviews
- Facebook → apify_facebook_scraper
- Instagram → apify_instagram_scraper
- TikTok → apify_tiktok_scraper
- Twitter/X → apify_twitter_scraper
- Google Play Store → get_playstore_reviews
- Wikipedia → get_wikipedia_text

## Búsqueda web
- Usa search_web para encontrar links de redes sociales
- Ejemplo: search_web("iPhone 15 review site:youtube.com")
- Ejemplo: search_web("iPhone 15 site:reddit.com")
"""
```

---

## Convenciones de Código

### Para cada Tool:
```python
# tools/youtube.py
from langchain.tools import tool
from config import YOUTUBE_API_KEY

@tool
def get_youtube_comments(query: str, max_results: int = 50) -> dict:
    """
    Obtiene comentarios de videos de YouTube relacionados con la búsqueda.
    
    Args:
        query: Término de búsqueda (ej: "iPhone 15 review")
        max_results: Máximo de comentarios a obtener (default 50)
    
    Returns:
        dict con {"success": bool, "data": [{"comment": str, "author": str, ...}], "count": int, "error": str}
    """
    try:
        # Implementación aquí
        return {"success": True, "data": comments, "count": len(comments), "error": None}
    except Exception as e:
        return {"success": False, "data": [], "count": 0, "error": str(e)}
```

### Lazy Loading de Modelos (EVITAR cold start):
```python
# Patrón para modelos pesados (BART, spaCy, PySentimiento)
_model = None

def _get_model():
    global _model
    if _model is None:
        from transformers import pipeline
        _model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _model
```

### Variables de Entorno:
```python
# config.py - ÚNICO lugar donde se cargan
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
# ... TODAS las demás
```

---

## .env.example

```env
# LLMs
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=optional_openai_key
OPENROUTER_API_KEY=optional_openrouter_key

# Search
SERPAPI_API_KEY=your_serpapi_key
GOOGLE_CSE_API_KEY=optional_google_cse_key
GOOGLE_CSE_CX=optional_google_cse_cx

# APIs Oficiales
YOUTUBE_API_KEY=your_youtube_api_key
MAPS_API_KEY=your_google_maps_api_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_user_agent

# APIFY
APIFY_API_KEY=your_apify_api_key

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_KEY=your_service_role_key

# Proxies (opcional)
OXYLABS_USER=optional_oxylabs_user
OXYLABS_PASS=optional_oxylabs_pass

# App
APP_ENV=development
LOG_LEVEL=INFO
```

---

## Schema SQL (supabase/schema.sql)

```sql
-- Habilitar extensiones
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Tabla unificada de comentarios
CREATE TABLE unified_comments (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id      UUID NOT NULL DEFAULT gen_random_uuid(),
    social_media    TEXT NOT NULL CHECK (social_media IN ('youtube','twitter','reddit','facebook','instagram','tiktok','google_maps','playstore','wikipedia','generic_web')),
    query           TEXT NOT NULL,
    url             TEXT,
    username        TEXT,
    comment         TEXT NOT NULL,
    rating          REAL,
    likes           INTEGER,
    post_date       TIMESTAMPTZ,
    category        TEXT,
    sentiment       TEXT CHECK (sentiment IN ('Positivo','Negativo','Neutral','Error')),
    emotion         TEXT CHECK (emotion IN ('Alegría','Tristeza','Enojo','Miedo','Sorpresa','Asco','Neutral','Error','Desconocida')),
    embedding       VECTOR(768),
    extraction_date TIMESTAMPTZ DEFAULT NOW(),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Índices
CREATE INDEX idx_unified_session ON unified_comments(session_id);
CREATE INDEX idx_unified_user ON unified_comments(user_id);
CREATE INDEX idx_unified_media ON unified_comments(social_media);
CREATE INDEX idx_unified_sentiment ON unified_comments(sentiment);
CREATE INDEX idx_unified_category ON unified_comments(category);

-- Índice pgvector para búsqueda semántica
CREATE INDEX idx_unified_embedding ON unified_comments 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Función RPC para búsqueda semántica
CREATE OR REPLACE FUNCTION match_comments(
    query_embedding VECTOR(768),
    p_session_id UUID,
    p_user_id UUID,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    comment TEXT,
    social_media TEXT,
    sentiment TEXT,
    emotion TEXT,
    category TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        uc.id,
        uc.comment,
        uc.social_media,
        uc.sentiment,
        uc.emotion,
        uc.category,
        1 - (uc.embedding <=> query_embedding) AS similarity
    FROM unified_comments uc
    WHERE uc.session_id = p_session_id
      AND uc.user_id = p_user_id
      AND uc.embedding IS NOT NULL
    ORDER BY uc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Chat history
CREATE TABLE chat_history (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id  UUID NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chat_session ON chat_history(session_id);
```

---

## RLS Policies (supabase/rls_policies.sql)

```sql
-- Habilitar RLS
ALTER TABLE unified_comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;

-- Políticas para unified_comments
CREATE POLICY "Users can view their own comments"
    ON unified_comments FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own comments"
    ON unified_comments FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Service role can manage all"
    ON unified_comments
    USING (true)
    WITH CHECK (true);

-- Políticas para chat_history
CREATE POLICY "Users can view their own chats"
    ON chat_history FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own chats"
    ON chat_history FOR INSERT
    WITH CHECK (auth.uid() = user_id);
```

---

## Lo que NO debes hacer

- ❌ NO uses Selenium/ChromeDriver → usa APIFY para FB, IG, TikTok, X
- ❌ NO crees múltiples tablas por plataforma → usa `unified_comments`
- ❌ NO uses Pinecone → pgvector en Supabase es suficiente
- ❌ NO implementes auth custom → usa Supabase Auth nativo
- ❌ NO cargues modelos en cada request → usa lazy loading global
- ❌ NO hagas scraping sin autorización → usa APIs oficiales o APIFY
- ❌ NO hardcodees API keys → siempre de `.env`
- ❌ NO uses `print()` para logs → usa `logging`
- ❌ NO retornes DataFrames directo de tools → usa dict estandarizado
- ❌ NO bloquees el UI durante scraping → usa generadores/yield en Gradio

---

## Testing en HuggingFace

```bash
# Local
pip install -r requirements.txt
python app.py

# HF Spaces (Docker)
docker build -t chismesitogpt .
docker run -p 7860:7860 --env-file .env chismesitogpt
```

La app debe escuchar en `0.0.0.0:7860` (puerto default de Gradio, también default de HF Spaces).

---

## Referencia rápida: Type Hints

```python
from typing import TypedDict, Optional

class CommentDict(TypedDict):
    comment: str
    social_media: str
    username: Optional[str]
    url: Optional[str]
    post_date: Optional[str]
    likes: Optional[int]
    rating: Optional[float]

class ToolResult(TypedDict):
    success: bool
    data: list[dict]
    count: int
    error: Optional[str]
```
