# ChismesitoGPT v2

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![UI Framework](https://img.shields.io/badge/Gradio-6.x-orange.svg)](https://gradio.app/)
[![Database](https://img.shields.io/badge/Supabase-PostgreSQL%20%2F%20pgvector-green.svg)](https://supabase.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**ChismesitoGPT v2** es una plataforma avanzada de inteligencia social y análisis de opiniones. Utilizando agentes orquestadores de LLMs (LangChain + Gemini/DeepSeek V4/Claude), la aplicación busca enlaces relevantes en tiempo real, extrae comentarios de múltiples redes sociales, clasifica y analiza su contenido (sentimientos, emociones y categorías auto-generadas) y almacena todo en una base de datos vectorial para ofrecer análisis descriptivo y un chat RAG (Retrieval-Augmented Generation) contextualizado.

> **Nota de reescritura:** Este repositorio contiene dos carpetas. `chismesito_gpt_old_version/` es la **versión anterior y ya obsoleta** (app Shiny monolítica con scraping basado en Selenium/ChromeDriver y Pinecone). `chismesito_gpt_nueva_version/` es la **versión mejorada y actual** documentada en este README. Toda la funcionalidad nueva vive en la carpeta `nueva_version`.

---

## Arquitectura de la Aplicación



![](./chismesito_gpt_nueva_version/icons/imagenes/Chismesito_gpt_architecture.png)


## Versión Anterior (Obsoleta)

La versión anterior, conocida internamente como **"Iris Social Media Downloader" (v0.5)**, era una aplicación construida con **Python Shiny** cuyo objetivo era descargar y analizar comentarios de redes sociales. En resumen:

* **Interfaz:** Aplicación Shiny (`app.py`, `app_module/ui.py`, `app_module/server.py`) con un único flujo monolítico de ~6,600 líneas de código.
* **Extracción:** Combinaba APIs oficiales (YouTube, Reddit, Twitter vía `tweepy`) con **scraping mediante Selenium + ChromeDriver** para Facebook, Instagram, TikTok, X y Google Maps (carpeta `scrapers/`, con `driver/chromedriver` incluido).
* **Análisis (NLP):** Sentimiento, emociones, temas y resúmenes con `spaCy`, `transformers` y `pysentimiento`.
* **Almacenamiento:** PostgreSQL vía Supabase para datos relacionales, y **Pinecone** como base de datos vectorial independiente (embeddings de Gemini, 768d) para búsqueda por similitud.
* **LLMs:** Gemini, OpenAI, OpenRouter y DeepSeek, orquestados de forma ad-hoc (sin agente).
* **Despliegue:** Previsto sobre AWS (ECS/Fargate + ALB + RDS + S3) debido a la dependencia de navegador headless (Selenium/Playwright).
* **Limitaciones:** Código poco modular, frágil ante cambios de los sitios (Selenium), servicio vectorial externo (Pinecone) y sin control de acceso ni chat conversacional sobre los datos.

---

## Qué hace ahora (Nueva Versión) vs. la Anterior

La nueva versión es una **reescritura completa** centrada en robustez, modularidad y una experiencia de usuario guiada. Las mejoras principales frente a la versión anterior son:

| Aspecto | Versión Anterior (Obsoleta) | Nueva Versión (ChismesitoGPT v2) |
|---------|----------------------------|----------------------------------|
| **Framework UI** | Python Shiny monolítico (~6,600 líneas) | **Gradio** con *Wizard* de 3 pasos (Configuración → Dashboard → Chat RAG) |
| **Orquestación** | Llamadas ad-hoc a LLMs | **Agente LangChain** (tool-calling) con Gemini/DeepSeek/Claude |
| **Scraping** | Selenium + ChromeDriver (frágil, requiere navegador) | **APIFY** para FB, IG, TikTok, X, Maps + **APIs oficiales** (YouTube, Reddit, Play Store) |
| **Base de datos vectorial** | Pinecone (servicio externo) | **pgvector dentro de Supabase** (mismo PostgreSQL, sin servicio extra) |
| **Embeddings** | Gemini 768d | Gemini **3072d** (`gemini-embedding-2`) con índices HNSW |
| **Acceso** | Sin autenticación | **Whitelist** (`allowed_users`) + **Supabase Auth** nativo |
| **Búsqueda** | Solo descriptiva | **Búsqueda semántica** + **Chat RAG** contextualizado |
| **Arquitectura** | Monolito poco modular | Modular: `tools/`, `pipeline/`, `db/`, `ui/` |
| **Despliegue** | AWS (por dependencia de navegador) | **Hugging Face Spaces** con Docker (ZeroGPU) |

**Resumen de mejoras:**
1. **Sin Selenium/ChromeDriver:** el scraping se delega a actores gestionados de APIFY y a APIs oficiales, eliminando la fragilidad y la necesidad de navegador headless.
2. **Todo-en-uno con Supabase:** datos relacionales y vectoriales (pgvector + HNSW) en un solo lugar, con RLS y auditoría (`request_performance`, `user_requests`).
3. **Experiencia guiada:** asistente de 3 pasos que separa configuración, visualización y chat.
4. **Chat RAG:** el usuario puede conversar directamente con sus comentarios usando búsqueda semántica, algo inexistente en la versión anterior.
5. **Control de acceso:** whitelist y autenticación nativa en lugar de acceso abierto.
6. **Mayor calidad de embeddings:** 3072 dimensiones para búsquedas semánticas más precisas.

---

## Explicación del Nuevo Desarrollo

La nueva versión se diseñó como una **aplicación agent-based**. En lugar de un flujo rígido, un **agente LangChain** decide qué *tools* usar según el prompt y las redes seleccionadas:

1. **Búsqueda de enlaces reales** con SerpAPI (más DuckDuckGo como respaldo), evitando inventar URLs.
2. **Extracción multi-plataforma** enrutada automáticamente: YouTube y Reddit vía APIs oficiales; Facebook, Instagram, TikTok, X y Google Maps vía APIFY; Play Store con scraper nativo.
3. **Pipeline de análisis** (`pipeline/analyzer.py`): el LLM genera 5–10 categorías a partir de una muestra del 20% de los comentarios, se clasifica el lote completo con un clasificador zero-shot (`mDeBERTa`/`BART`), se detectan sentimiento y emoción en español con `PySentimiento` (RoBERTuito) y se generan embeddings con Gemini.
4. **Persistencia vectorial** en Supabase (pgvector, índice HNSW) mediante la función RPC `match_comments` para búsqueda semántica.
5. **Interfaz progresiva Gradio** que despliega datos, dashboard de Plotly y chat RAG conforme avanza el proceso, con exportación CSV/Excel.

El resultado es una plataforma más mantenible, desplegable en Hugging Face Spaces y con capacidades conversacionales que la versión anterior no poseía.

---

## Interfaz de la Aplicación


---

## Características Principales

* **Whitelist de Acceso & Autenticación Nativa:** Control de acceso estricto mediante whitelist en base de datos (`allowed_users`) y login integrado a Supabase Auth.
* **Wizard de 3 Pasos (Stitch UI):**
    1. **Paso 1 (Configuración):** Ajuste de prompts, selección visual de redes sociales, configuración geográfica para Google Maps y modo de extracción (Manual o Automático).
    2. **Paso 2 (Dashboard):** Visualizaciones de análisis de sentimiento, distribución de emociones, categorías más populares y mapas interactivos.
    3. **Paso 3 (Chat RAG):** Chat directo con los comentarios usando búsquedas semánticas mediante base de datos vectorial.
* ** Orquestación de Scrapers Multi-Plataforma:**
  * **YouTube:** API oficial Data v3.
  * **Reddit:** PRAW (Python Reddit API Wrapper).
  * **Google Play Store:** Scraper nativo.
  * **Facebook, Instagram, TikTok, X (Twitter), Google Maps:** Integración robusta con actores y scrapers de **APIFY** (sin depender de Selenium local o ChromeDriver).
* **Pipeline de NLP de Alta Precisión:**
  * **Clasificación Temática:** El LLM genera dinámicamente de 5 a 10 categorías basadas en una muestra representativa (20% de comentarios) y clasifica todo el lote utilizando el modelo zero-shot `BART-large-mnli`.
  * **Sentimiento y Emoción:** Detección en español nativo con la suite `PySentimiento`.
  * **Embeddings Vectoriales:** Generación de vectores de **3072 dimensiones** en tiempo real mediante `gemini-embedding-2`.
* **Búsqueda Semántica Integrada (pgvector + HNSW):** Indexación ultrarrápida usando índices de grafo jerárquico (`hnsw` con cast a `halfvec`) dentro del esquema relacional de Supabase.
* **Auditoría & Logs:** Tablas de rendimiento y profiling (`request_performance`) y log de peticiones del usuario (`user_requests`) para control de costos e historial de ejecución.

* **Migración a AWS:** Se plantea una migración a AWS en el futuro cercano para poder ejecutarla en un entorno más controlado, se planea también migrar el desarrollo a un backend más robusto (Django) y un frontend más dinámico (React).


---

## Arquitectura de Base de datos

[Link al diagrama de arquitectura de base de datos](https://dbdiagram.io/d/chismesito_gpt_diagram-6a4c1b0636d348d1207ecab5)

![](./chismesito_gpt_nueva_version/icons/imagenes/chismesito_gpt_diagram.png)


## Arquitectura de Archivos

```text
chismesito_gpt_nueva_version/
├── app.py                    # Entry point Gradio & Layout Wizard
├── config.py                 # Configuración central y carga de variables de entorno
├── llm_manager.py            # Orquestador multi-provider (Gemini, DeepSeek, Claude)
├── requirements.txt          # Dependencias de Python
├── supabase/
│   ├── schema.sql            # Definición de tablas, índices HNSW y función RPC
│   ├── rls_policies.sql      # Row Level Security y whitelist de accesos
│   └── migration_v2_embed3072.sql # Migración segura (para DBs con datos activos)
├── tools/
│   ├── search_tool.py        # SerpAPI para búsqueda de enlaces
│   ├── youtube_tool.py       # API oficial de YouTube
│   ├── reddit_tool.py        # API oficial de Reddit
│   ├── apify_tool.py         # Conector APIFY (X, FB, IG, TikTok, Maps)
│   ├── sentiment_tool.py     # Análisis de sentimiento (PySentimiento)
│   ├── emotion_tool.py       # Análisis de emociones (PySentimiento)
│   ├── categories_tool.py    # Generación de categorías temáticas con LLM
│   ├── zero_shot_tool.py     # Clasificación BART-large-mnli
│   ├── embeddings_tool.py    # Generación de embeddings con Gemini (3072d)
│   └── export_tool.py        # Exportador CSV/Excel
├── pipeline/
│   ├── orchestrator.py       # Orquestador del flujo completo
│   ├── analyzer.py           # Pipeline secuencial de enriquecimiento
│   └── rag.py                # Recuperación semántica y chat
├── db/
│   ├── supabase_client.py    # Cliente singleton de Supabase
│   ├── ops.py                # CRUD de comentarios y logs
│   └── vector.py             # Operaciones vectoriales
└── ui/
    ├── styles.py             # CSS global y maquetado de componentes HTML
    ├── dashboard.py          # Definición de gráficos en Plotly
    └── app_flujo.txt         # Documentación detallada del flujo paso a paso
```

---

## Variables de Entorno (`.env`)

Crea un archivo `.env` en la raíz del proyecto tomando como referencia el siguiente esquema:

```env
# LLMs - APIs oficiales
GEMINI_API_KEY=tu_gemini_api_key
DEEPSEEK_API_KEY=tu_deepseek_api_key
ANTHROPIC_API_KEY=tu_anthropic_api_key

# Búsqueda
SERPAPI_API_KEY=tu_serpapi_api_key

# APIs Oficiales de Redes
YOUTUBE_API_KEY=tu_youtube_api_key
MAPS_API_KEY=tu_google_maps_api_key
MAPBOX_TOKEN=tu_mapbox_token
REDDIT_CLIENT_ID=tu_reddit_client_id
REDDIT_CLIENT_SECRET=tu_reddit_client_secret
REDDIT_USER_AGENT=chismesito_gpt_v2/1.0

# APIFY (Facebook, Instagram, TikTok, Twitter/X, Google Maps Scraper)
APIFY_API_KEY=tu_apify_api_key

# Supabase
SUPABASE_URL=https://tu-proyecto.supabase.co
SUPABASE_ANON_KEY=tu_anon_key
SUPABASE_SERVICE_KEY=tu_service_role_key

# Configuración de Aplicación
APP_ENV=development
LOG_LEVEL=INFO
ALLOWED_EMAILS=admin@empresa.com,usuario@empresa.com
```

---

## Instalación y Despliegue Local

### 1. Clonar el repositorio y preparar el entorno conda

```bash
conda create -n chismesito_gpt python=3.11 -y
conda activate chismesito_gpt
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_md
```

### 3. Configurar Base de Datos en Supabase

1. Ingresa a tu proyecto en Supabase.
2. Ve a **Database → Extensions** y activa las extensiones `vector` y `uuid-ossp`.
3. Abre el **SQL Editor** y ejecuta en orden:
   * El archivo [`supabase/schema.sql`](./supabase/schema.sql) (crea las tablas e índices HNSW).
   * El archivo [`supabase/rls_policies.sql`](./supabase/rls_policies.sql) (aplica seguridad e inserta las cuentas de correo permitidas).
   *(Si estás migrando una base de datos con datos de comentarios existentes, ejecuta [`supabase/migration_v2_embed3072.sql`](./supabase/migration_v2_embed3072.sql) para actualizar los embeddings a 3072d sin perder información).*

### 4. Lanzar la aplicación

```bash
python app.py
```

Abre en tu navegador la dirección `http://localhost:7860`.

---

## Despliegue en Hugging Face Spaces

La aplicación se ejecuta nativamente en entornos Hugging Face mediante Docker.

### Estructura de Autenticación en Despliegue

```text
Cliente (Gradio)                    Servidor (Supabase)
       │                                     │
       │── Email + Password ──────────────►  │ Auth.sign_in()
       │◄── JWT Access Token ──────────────  │
       │                                     │
       │  [Token en gr.State]                │
       │                                     │
       │── run_pipeline() ────────────────►  │ DB CRUD (Service Key)
       │   (JWT verificado)                  │ [Aplica RLS en base al id del JWT]
```

### Flujo de Variables de Entorno en HF

1. En la configuración de tu Space, añade las variables listadas en el apartado `.env` como **Secrets**.
2. Asegúrate de configurar la variable `PORT` o utilizar el puerto de escucha automático que asigna Hugging Face.

---

## Flujo de Datos Detallado

Para un desglose completo de cómo se procesa la información desde que el usuario introduce un término de búsqueda hasta que se genera la respuesta en el chat RAG, consulta la guía interna [`ui/app_flujo.txt`](./ui/app_flujo.txt).


## Pasos a futuro

* Límite de caracteres por prompt
* El usuario podrá ver sus sesiones pasadas
* Un usuario administrador para poder gestionar los usuarios permitidos y el límite de creditos por usuario
* Límite de 2 USD por usuario por mes
* Animaciones de carga y progreso para cada paso del wizard
* Notificaciones de navegador
* Responsivo para uso en móvil
* Prompts por voz usando Whisper como herramienta de transcripción
