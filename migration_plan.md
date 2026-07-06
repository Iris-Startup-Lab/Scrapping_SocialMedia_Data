# Plan de Migración: ChismesitoGPT v0.5 → v1.0 (Agent-Based)

## Resumen Ejecutivo

Migrar la actual app Shiny Python a una arquitectura **basada en agentes con Gradio + LangChain**, desplegable en HuggingFace Spaces para testing, con ruta de migración futura a React + FastAPI para producción en AWS.

---

## 1. Diagnóstico de la Arquitectura Actual (v0.5)

### Lo que funciona
| Componente | Estado | Problema |
|-----------|--------|---------|
| Scraping YouTube, Reddit, Maps (APIs oficiales) | Funcional | Código duplicado entre `data_fetchers.py` y `scrapers.py` |
| Selenium + Chrome para Facebook | Funcional | Pesado, frágil, dependencia de ChromeDriver |
| Análisis de sentimiento/emociones | Funcional | Modelos cargados en cada sesión (cold start ~30s) |
| PostgreSQL (Supabase) | Funcional | 7 tablas separadas, esquema no unificado |
| Pinecone Vector DB | Funcional | Overkill: Supabase pgvector bastaría |
| Shiny UI | Funcional | UI legacy, sin diseño moderno, login básico |

### Deuda técnica crítica
1. **Sin separación de capas** — app.py de 2000+ líneas, todo acoplado
2. **Sin autenticación robusta** — login por email/empleado en tabla custom
3. **Tablas separadas por plataforma** — difícil para análisis cross-platform
4. **Sin sistema de agentes** — flujo secuencial rígido
5. **ChromeDriver bundling** — incompatible con contenedores ligeros

---

## 2. Arquitectura Objetivo (v1.0)

```
┌──────────────────────────────────────────────────────────────────┐
│                         USUARIO                                   │
│                  Input: "iPhone 15 opiniones"                     │
│                  + Selecciona: [YouTube] [Twitter] [Reddit]       │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                   GRADIO UI (Frontend)                            │
│  - Input único de prompt + sub-menú de redes con iconos          │
│  - Dashboard progresivo (secciones aparecen conforme avanza)      │
│  - Chat RAG sobre los datos                                      │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│               AGENTE ORQUESTADOR (LangChain Agent)                │
│  Modelo: Gemini 2.0 Flash / DeepSeek                             │
│  Tareas:                                                         │
│  1. Interpretar prompt del usuario                               │
│  2. Decidir qué tools llamar (YouTube, Reddit, APIFY...)         │
│  3. Recibir resultados y orquestar análisis                      │
│  4. Generar categorías (5-10) desde muestra de 20%               │
│  5. Responder preguntas RAG del usuario                          │
└───────┬───────┬───────┬───────┬───────┬──────────────────────────┘
        │       │       │       │       │
        ▼       ▼       ▼       ▼       ▼
┌──────┐ ┌────┐ ┌────┐ ┌────┐ ┌────────────┐
│Search│ │API │ │API │ │APIFY│ │Small Models│
│Tool  │ │YT  │ │Red.│ │Tool │ │            │
│      │ │    │ │    │ │     │ │BART (NLP)  │
│SerpAPI│ │Off.│ │Off.│ │FB,IG│ │Zero-Shot   │
│Google │ │    │ │    │ │TT,X │ │Classifier  │
│Search │ │    │ │    │ │     │ │            │
└──┬───┘ └──┬─┘ └──┬─┘ └──┬──┘ └─────┬──────┘
   │        │      │      │           │
   ▼        ▼      ▼      ▼           ▼
┌──────────────────────────────────────────────────────────────────┐
│                CAPA DE DATOS                                      │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │   Supabase       │  │   Vector Store   │  │   File Output  │  │
│  │   (PostgreSQL)   │  │   (pgvector)     │  │   (CSV/XLSX)   │  │
│  │                  │  │                  │  │                │  │
│  │ Tabla unificada: │  │ Chunks de        │  │ Descarga al    │  │
│  │ unified_comments │  │ comentarios      │  │ finalizar      │  │
│  │ + auth.users     │  │ + embeddings     │  │                │  │
│  └──────────────────┘  └──────────────────┘  └────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Stack Tecnológico Propuesto

| Capa | v0.5 (Actual) | v1.0 (Objetivo) | Razón |
|------|---------------|------------------|-------|
| **Framework UI** | Shiny Python | **Gradio** | HuggingFace-native, bloques progresivos, mejor para demos IA |
| **Orquestación** | Ninguna (secuencial) | **LangChain / LangGraph** | Agentes, tools, chains, memoria |
| **LLM Principal** | Gemini (genai) | Gemini 2.0 Flash + DeepSeek | Gemini gratis en HF, DeepSeek como fallback |
| **Search** | Ninguno | **SerpAPI + Google Custom Search** | Encontrar links de redes sociales |
| **Scraping APIs Oficiales** | Tweepy, googleapiclient, PRAW | Se mantienen (vía LangChain Tools) | Probados y funcionales |
| **Scraping No-Oficial** | Selenium + RapidAPI | **APIFY actors** (Facebook, TikTok, Instagram, X) | Más mantenible, sin ChromeDriver |
| **NLP Ligero** | PySentimiento, spaCy | Se mantienen | Rápidos, sin GPU |
| **Zero-Shot** | BART (transformers) | BART o DeBERTa-v3 | Clasificación de categorías |
| **Base de Datos** | Supabase 7 tablas + Pinecone | **Supabase (tabla única) + pgvector** | Simplifica esquema, unifica vector y relacional |
| **Auth** | Tabla custom `iris_email_employees_enabled` | **Supabase Auth** (email/password) | Robusto, RLS, OAuth listo |
| **Exportación** | openpyxl, xlsxwriter | Se mantienen | CSV y Excel |
| **Cache** | Archivos locales | **Supabase Storage** o `/tmp` en HF | Stateless en producción |
| **Proxy** | Oxylabs | Se mantiene opcional | Anti-bot para scraping |
| **Deploy Actual** | Local | **HuggingFace Spaces (Docker)** | => Free tier, GPU opcional |
| **Deploy Futuro** | — | **AWS ECS + RDS + ElastiCache** | Producción con React frontend |

---

## 4. Esquema de Base de Datos Unificado

### Tabla: `unified_comments` (Supabase PostgreSQL)

```sql
CREATE TABLE unified_comments (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID REFERENCES auth.users(id) NOT NULL,
  session_id      UUID NOT NULL,
  social_media    TEXT NOT NULL,   -- 'youtube', 'twitter', 'reddit', 'facebook', 'instagram', 'tiktok', 'google_maps', 'playstore'
  query           TEXT NOT NULL,   -- Prompt original del usuario
  url             TEXT,            -- URL de origen
  username        TEXT,            -- Nombre del autor
  comment         TEXT NOT NULL,   -- Texto del comentario/post
  rating          REAL,            -- Calificación (si aplica)
  likes           INTEGER,         -- Likes/interacciones
  post_date       TIMESTAMPTZ,     -- Fecha original del post
  category        TEXT,            -- Categoría asignada por zero-shot
  sentiment       TEXT,            -- 'Positivo', 'Negativo', 'Neutral'
  emotion         TEXT,            -- 'Alegría', 'Tristeza', 'Enojo', 'Miedo', 'Sorpresa', 'Asco', 'Neutral'
  extraction_date TIMESTAMPTZ DEFAULT NOW(),
  embedding       VECTOR(768)      -- Embedding Gemini para RAG (pgvector)
);

-- Índices
CREATE INDEX idx_unified_session ON unified_comments(session_id);
CREATE INDEX idx_unified_user ON unified_comments(user_id);
CREATE INDEX idx_unified_media ON unified_comments(social_media);
CREATE INDEX idx_embedding ON unified_comments USING ivfflat (embedding vector_cosine_ops);
```

### Auth: Supabase Auth (nativo)
- `auth.users` — tabla gestionada por Supabase
- Login: email + password (con confirmación opcional)
- RLS (Row Level Security) para aislar datos por usuario

### Ventajas sobre v0.5:
- **1 tabla vs 7 tablas** — queries cross-platform triviales
- **pgvector** — elimina dependencia de Pinecone
- **Auth nativo** — elimina tabla custom de login
- **RLS** — seguridad a nivel fila automática

---

## 5. Plan de Fases

### Fase 0: Preparación (1-2 días)

| Tarea | Descripción |
|-------|-------------|
| **0.1** | Crear repo nuevo o branch `v1-migration` |
| **0.2** | Configurar proyecto Supabase nuevo con pgvector |
| **0.3** | Activar Supabase Auth (email/password) |
| **0.4** | Crear tablas: `unified_comments`, `chat_history`, `user_sessions` |
| **0.5** | Configurar `.env` con API keys para v1 |
| **0.6** | Instalar stack: gradio, langchain, langgraph, langchain-community, supabase-py con pgvector |

### Fase 1: Tools del Agente (2-3 días)

Crear LangChain Tools modulares, cada una en su propio archivo:

```
tools/
├── __init__.py
├── search_tool.py          # SerpAPI + Google Custom Search → buscar links
├── youtube_tool.py         # YouTube Data API v3 → comentarios
├── reddit_tool.py          # PRAW → comentarios
├── google_maps_tool.py     # Google Maps API → reseñas
├── apify_tool.py           # APIFY actors (FB, IG, TikTok, X)
├── playstore_tool.py       # google-play-scraper → reseñas
├── wikipedia_tool.py       # Wikipedia → texto
├── web_scraper_tool.py     # Scraping genérico (requests + BS4 + proxy)
├── sentiment_tool.py       # PySentimiento + spaCy → sentimiento
├── emotion_tool.py         # PySentimiento → emociones
├── category_generator.py   # LLM genera 5-10 categorías
├── zero_shot_tool.py       # BART/DeBERTa → clasificador zero-shot
├── embedding_tool.py       # Gemini embeddings → vectorizar comentarios
└── export_tool.py          # CSV / XLSX download
```

Cada tool debe:
- Ser una función con `@tool` de LangChain
- Tener descripción clara para que el agente decida cuándo usarla
- Manejar errores con gracia
- Retornar resultados en formato estandarizado

### Fase 2: Agente Orquestador (2 días)

Archivo: `agent.py`

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Prompt del sistema para el agente
SYSTEM_PROMPT = """
Eres ChismesitoGPT, un asistente que extrae y analiza comentarios de redes sociales.

Tu workflow:
1. Recibes un prompt del usuario + redes sociales seleccionadas
2. Para cada red social, obtienes comentarios usando la tool adecuada
3. Consolidas todos los comentarios en un solo dataset
4. Generas 5-10 categorías con una muestra del 20% de comentarios
5. Clasificas cada comentario en una categoría
6. Analizas sentimiento y emoción de cada comentario
7. Vectorizas los comentarios para búsqueda semántica
8. Generas un dashboard resumen

Disponible para buscar en la web:
- SerpAPI o Google Custom Search para encontrar links relevantes
- NO inventes URLs, siempre usa las tools de búsqueda

Sé conciso. Muestra progreso al usuario en cada paso.
"""

tools = [
    search_web_tool,
    get_youtube_comments_tool,
    get_reddit_comments_tool,
    get_google_maps_reviews_tool,
    apify_scraper_tool,
    analyze_sentiment_tool,
    analyze_emotion_tool,
    generate_categories_tool,
    zero_shot_classify_tool,
    embed_comments_tool,
    export_to_csv_tool,
    export_to_xlsx_tool,
]
```

### Fase 3: UI con Gradio (2-3 días)

Archivo: `app_v1.py`

```python
import gradio as gr

with gr.Blocks(theme=gr.themes.Soft(), title="ChismesitoGPT v1") as demo:
    # Estado de sesión
    session_state = gr.State({})
    
    # Componentes en orden de aparición
    gr.Markdown("# 🕵️ ChismesitoGPT v1")
    
    # Paso 1: Input + selección de redes
    with gr.Group():
        prompt = gr.Textbox(label="¿Qué quieres investigar?", 
                           placeholder="Ej: Opiniones del iPhone 15 en México...")
        
        social_medias = gr.CheckboxGroup(
            choices=["YouTube", "Twitter/X", "Reddit", "Facebook", 
                    "Instagram", "TikTok", "Google Maps"],
            label="Redes sociales",
            interactive=True
        )
        search_btn = gr.Button("🔍 Buscar y analizar", variant="primary")
    
    # Paso 2: Resultados progresivos (inicialmente ocultos)
    with gr.Column(visible=False) as results_col:
        gr.Markdown("## 📊 Resultados")
        search_status = gr.Markdown()
        
        # Tablas de datos
        with gr.Tab("Datos"):
            comments_table = gr.Dataframe(label="Comentarios recolectados")
            download_csv = gr.File(label="Descargar CSV")
            download_xlsx = gr.File(label="Descargar Excel")
        
        # Dashboard
        with gr.Tab("Dashboard"):
            sentiment_plot = gr.Plot(label="Sentimiento")
            emotion_plot = gr.Plot(label="Emociones")
            category_plot = gr.Plot(label="Categorías")
        
        # Chat RAG
        with gr.Tab("Chat con los datos"):
            chatbot = gr.Chatbot(label="Pregunta sobre los datos")
            chat_input = gr.Textbox(label="Tu pregunta")
            chat_btn = gr.Button("Preguntar")
    
    # Login
    with gr.Accordion("🔐 Login", open=False):
        email = gr.Textbox(label="Email")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Iniciar Sesión")
        login_status = gr.Markdown()
```

**Comportamiento progresivo:**
1. Usuario escribe prompt + selecciona redes
2. Al hacer clic en "Buscar", se muestran las secciones conforme se completan
3. Primero: status "Buscando en YouTube... 45 comentarios encontrados"
4. Luego: tabla de datos
5. Luego: gráficos de dashboard
6. Finalmente: chat RAG habilitado

### Fase 4: Pipeline de Análisis (1-2 días)

```
Comentarios crudos (DataFrame)
         │
         ▼
┌─────────────────────────────┐
│ 1. LLM genera 5-10 categorías│  ← Muestra 20% de comentarios
│    (Gemini / DeepSeek)       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 2. Zero-Shot Classification  │  ← Clasifica CADA comentario
│    (BART-large-mnli)         │     en las categorías generadas
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 3. Sentimiento + Emoción     │  ← PySentimiento (rápido, español)
│    (por cada comentario)     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 4. Vectorización (RAG)       │  ← Gemini embeddings (768d)
│    (por cada comentario)     │     guardar en pgvector
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 5. Guardar en Supabase       │  ← unified_comments table
│    + Exportar CSV/XLSX       │
└─────────────────────────────┘
```

### Fase 5: RAG sobre Comentarios (1 día)

```python
# rag.py
from supabase import create_client

def semantic_search(query: str, session_id: str, user_id: str, top_k: int = 10):
    """
    1. Embeddear la query del usuario con Gemini
    2. Buscar en pgvector los top_k comentarios más cercanos
    3. Retornar contexto para el LLM
    """
    embedding = embed_query_gemini(query)
    
    # Búsqueda en pgvector
    results = supabase.rpc(
        'match_comments',
        {
            'query_embedding': embedding,
            'session_id': session_id,
            'user_id': user_id,
            'match_count': top_k
        }
    ).execute()
    
    context = "\n".join([r['comment'] for r in results.data])
    return context

def rag_chat(user_question: str, session_id: str, user_id: str):
    """Responde preguntas usando RAG sobre los comentarios de la sesión."""
    context = semantic_search(user_question, session_id, user_id)
    
    prompt = f"""Contexto de comentarios:
{context}

Pregunta del usuario: {user_question}

Responde basándote SOLO en el contexto proporcionado. Si no hay suficiente información, dilo."""
    
    return llm.invoke(prompt)
```

### Fase 6: Auth (1 día)

Usar **Supabase Auth** nativo en lugar de la tabla custom actual:

```python
# auth.py
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def sign_up(email: str, password: str):
    return supabase.auth.sign_up({"email": email, "password": password})

def sign_in(email: str, password: str):
    return supabase.auth.sign_in_with_password({"email": email, "password": password})

def sign_out(access_token: str):
    return supabase.auth.sign_out()

def get_user(access_token: str):
    return supabase.auth.get_user(access_token)
```

### Fase 7: Deploy en HuggingFace (1 día)

1. Crear `Dockerfile` para HuggingFace Spaces:
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y curl
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app_v1.py"]
```

2. Crear `requirements.txt` para v1:
```
gradio>=4.0.0
langchain>=0.2.0
langchain-google-genai
langchain-community
langgraph
supabase
pgvector
psycopg2-binary
google-generativeai
google-api-python-client
googlemaps
praw
tweepy
pysentimiento
transformers
torch
pandas
numpy
plotly
matplotlib
seaborn
python-dotenv
openai
openpyxl
xlsxwriter
google-search-results
deep-translator
aiohttp
httpx
```

3. Subir a HuggingFace Space (tipo Docker):
```bash
git init
git add .
git commit -m "v1 migration"
git remote add origin https://huggingface.co/spaces/USER/chismesitogpt-v1
git push origin main
```

---

## 6. Diagrama de Secuencia

```
Usuario          Gradio UI       Agente          Tools           Supabase
  │                 │              │               │                │
  │─prompt+redes───►│              │               │                │
  │                 │─orquestar───►│               │                │
  │                 │              │─search_links──►│                │
  │                 │              │◄───links───────│                │
  │                 │              │                           │
  │                 │              │─youtube_api───►│               │
  │                 │              │◄───comments────│               │
  │                 │              │                           │
  │                 │              │─reddit_api────►│               │
  │                 │              │◄───comments────│               │
  │                 │              │                           │
  │                 │              │─apify_scrape──►│               │
  │                 │              │◄───comments────│               │
  │                 │              │                           │
  │                 │              │─generate_categories──►│        │
  │                 │              │◄───categories───│               │
  │                 │              │                           │
  │                 │              │─zero_shot────────►│          │
  │                 │              │◄───classified────│          │
  │                 │              │                           │
  │                 │              │─sentiment_emotion──►│         │
  │                 │              │◄───analyzed───────│         │
  │                 │              │                           │
  │                 │              │─embed────────────►│          │
  │                 │              │◄───vectors────────│          │
  │                 │              │                           │
  │                 │              │──────save_all──────────────────►│
  │                 │◄──dashboard──│                           │
  │◄──ver───────────│              │                           │
  │                 │              │                           │
  │─chat_question──►│──────────────►│                           │
  │                 │              │──search_vectors──────────────►│
  │                 │              │◄───context───────────────────│
  │                 │              │─generate_response──►│          │
  │                 │◄──answer─────│                           │
  │◄──respuesta─────│              │                           │
```

---

## 7. Métricas de Éxito

| Métrica | v0.5 | v1.0 Target |
|---------|------|-------------|
| Tiempo de cold start | ~45s (carga de modelos) | <10s (lazy load + caché) |
| Plataformas soportadas | 8 (4 con scraping frágil) | 10 (via APIFY + APIs) |
| Tablas BD | 7 separadas | 1 unificada |
| Auth | Custom SQL table | Supabase Auth (estándar) |
| Chat RAG | Manual (Pinecone) | Automático (pgvector) |
| UI | Shiny (legacy) | Gradio (moderno, progresivo) |
| Deploy | Solo local | HuggingFace + AWS-ready |
| Mantenibilidad | 2000+ líneas en 1 archivo | Módulos por responsabilidad |

---

## 8. Timeline Estimado

| Fase | Días | Entregable |
|------|------|-----------|
| Fase 0: Preparación | 1-2 | Repo, Supabase config, .env |
| Fase 1: Tools | 2-3 | 14+ LangChain tools funcionales |
| Fase 2: Agente | 2 | AgentExecutor orquestando tools |
| Fase 3: UI Gradio | 2-3 | Interfaz progresiva con login |
| Fase 4: Pipeline | 1-2 | Análisis end-to-end |
| Fase 5: RAG | 1 | Chat semántico sobre datos |
| Fase 6: Auth | 1 | Supabase Auth integrado |
| Fase 7: Deploy HF | 1 | Docker en HuggingFace Spaces |
| **Total** | **11-15 días** | v1.0 funcional en producción (HF) |
