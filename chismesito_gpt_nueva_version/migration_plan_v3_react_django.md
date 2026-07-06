# Plan de Migración: Gradio → React + Django (ChismesitoGPT v3)

## Objetivo

Migrar ChismesitoGPT de una app monolítica Gradio/Python a una arquitectura **React (frontend) + Django REST Framework (backend)**, manteniendo Supabase como base de datos y pgvector como motor vectorial.

---

## 1. Diagnóstico: v2 (Gradio) → v3 (React + Django)

### Lo que se pierde con Gradio

| Debilidad de Gradio | Cómo lo resuelve React |
|---------------------|----------------------|
| UI limitada a bloques predefinidos | Componentes custom (Tailwind, shadcn) |
| Sin ruteo real (todo en una página) | React Router → navegación SPA real |
| Difícil manejar estado complejo | Zustand / Redux / Context API |
| Estilo CSS limitado | TailwindCSS, animaciones, diseño responsivo |
| Difícil integrar auth avanzada | Supabase Auth JS client nativo |
| No escala a muchos usuarios concurrentes | Frontend estático + API REST escalable |
| Cold start pesado (~30s en HF Spaces) | Frontend carga instantáneo (CDN) |

### Lo que se mantiene

| Componente | Migración |
|-----------|-----------|
| Supabase PostgreSQL + pgvector | Sin cambios |
| Tools (YouTube, Reddit, APIFY, etc.) | Se migran a servicios Django |
| Pipeline de análisis | Se migra a Django tasks/jobs |
| APIs externas (Gemini, Claude, DeepSeek) | Igual, desde backend Django |
| Modelo NLP (PySentimiento, BART) | Igual, desde backend |
| Embeddings (Gemini text-embedding-004) | Igual, desde backend |

---

## 2. Arquitectura v3

```
┌──────────────────────────────────────────────────────┐
│                    CLIENTE                            │
│  ┌────────────────────────────────────────────┐      │
│  │         React 18 + Vite + TypeScript       │      │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────┐ │      │
│  │  │ Tailwind │ │ Zustand  │ │ React Router│ │      │
│  │  │   CSS    │ │  (state) │ │   (rutas)  │ │      │
│  │  └──────────┘ └──────────┘ └────────────┘ │      │
│  │  ┌──────────────────────────────────┐     │      │
│  │  │  Supabase Auth JS (login/signup) │     │      │
│  │  └──────────────────────────────────┘     │      │
│  └────────────────────────────────────────────┘      │
│                         │                             │
│                    HTTPS (REST)                       │
└─────────────────────────┼───────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────┐
│                    BACKEND                             │
│  ┌────────────────────────────────────────────┐      │
│  │           Django + DRF (Python 3.12)        │      │
│  │                                              │      │
│  │  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │      │
│  │  │  Auth    │ │  Scraping│ │  Analysis    │ │      │
│  │  │  (JWT)   │ │  (tasks) │ │  (pipeline)  │ │      │
│  │  └──────────┘ └──────────┘ └─────────────┘ │      │
│  │  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │      │
│  │  │  Chat    │ │  Export  │ │  Dashboard   │ │      │
│  │  │  (RAG)   │ │  (CSV)   │ │  (metrics)   │ │      │
│  │  └──────────┘ └──────────┘ └─────────────┘ │      │
│  │  ┌──────────┐ ┌──────────────────────────┐ │      │
│  │  │  Celery  │ │  LLM Manager (multi-API) │ │      │
│  │  │  (async) │ │  Gemini/Claude/DeepSeek  │ │      │
│  │  └──────────┘ └──────────────────────────┘ │      │
│  └────────────────────────────────────────────┘      │
│                         │                             │
│                    ┌────┴────┐                        │
│                    │ Redis   │ (cache + Celery broker)│
│                    └─────────┘                        │
└─────────────────────────┼───────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────┐
│                    DATOS                               │
│  ┌──────────────────┐ ┌──────────────────────┐      │
│  │  Supabase        │ │  S3 / Cloudflare R2  │      │
│  │  PostgreSQL      │ │  (assets, exports)   │      │
│  │  + pgvector      │ │                      │      │
│  └──────────────────┘ └──────────────────────┘      │
└──────────────────────────────────────────────────────┘
```

---

## 3. Estructura de Archivos v3

```
chismesitogpt_v3/
├── backend/                          # Django project
│   ├── manage.py
│   ├── requirements.txt
│   ├── config/                       # Django settings
│   │   ├── __init__.py
│   │   ├── settings.py               # Base + env loading
│   │   ├── urls.py                   # Root URL conf
│   │   ├── wsgi.py
│   │   └── asgi.py
│   ├── apps/
│   │   ├── auth_app/                 # Supabase Auth + JWT
│   │   │   ├── models.py
│   │   │   ├── views.py
│   │   │   ├── serializers.py
│   │   │   └── urls.py
│   │   ├── scraping/                 # Orquestador de scraping
│   │   │   ├── services/
│   │   │   │   ├── youtube.py        # YouTube API
│   │   │   │   ├── reddit.py         # PRAW
│   │   │   │   ├── playstore.py      # Play Store
│   │   │   │   ├── apify.py          # APIFY actors
│   │   │   │   └── search.py         # SerpAPI
│   │   │   ├── tasks.py              # Celery tasks
│   │   │   ├── views.py
│   │   │   ├── serializers.py
│   │   │   └── urls.py
│   │   ├── analysis/                 # Pipeline de análisis
│   │   │   ├── services/
│   │   │   │   ├── sentiment.py
│   │   │   │   ├── emotion.py
│   │   │   │   ├── categories.py
│   │   │   │   ├── zero_shot.py
│   │   │   │   └── embeddings.py
│   │   │   ├── tasks.py
│   │   │   └── views.py
│   │   ├── chat/                     # Chat RAG
│   │   │   ├── services/
│   │   │   │   └── rag.py
│   │   │   ├── views.py
│   │   │   └── urls.py
│   │   ├── dashboard/                # Métricas agregadas
│   │   │   ├── views.py
│   │   │   └── urls.py
│   │   └── export/                   # CSV/Excel export
│   │       ├── views.py
│   │       └── urls.py
│   ├── core/                         # Lógica compartida
│   │   ├── llm_manager.py           # Gemini/Claude/DeepSeek
│   │   ├── supabase_client.py       # Singleton DB
│   │   ├── pagination.py
│   │   └── permissions.py
│   └── celery_app.py                # Celery config
│
├── frontend/                         # React project
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── public/
│   │   └── favicon.ico
│   └── src/
│       ├── main.tsx                  # Entry point
│       ├── App.tsx                   # Root component + Router
│       ├── api/
│       │   └── client.ts            # Axios instance + interceptors
│       ├── auth/
│       │   ├── LoginPage.tsx
│       │   ├── SignUpPage.tsx
│       │   └── useAuth.ts           # Auth hook (Supabase JS)
│       ├── search/
│       │   ├── SearchPage.tsx        # Input + selector de redes
│       │   ├── SocialMediaPicker.tsx # Componente con iconos
│       │   └── useSearch.ts         # Hook de búsqueda
│       ├── results/
│       │   ├── ResultsPage.tsx       # Tabla + gráficos + export
│       │   ├── DataTable.tsx
│       │   ├── DashboardCharts.tsx   # Recharts/Plotly.js
│       │   └── ExportButtons.tsx
│       ├── chat/
│       │   ├── ChatPage.tsx          # Chat RAG
│       │   └── ChatBubble.tsx
│       ├── components/
│       │   ├── Layout.tsx            # Navbar + Sidebar
│       │   ├── ProgressBar.tsx       # Scraping progress
│       │   ├── ModelSelector.tsx     # Dropdown modelos IA
│       │   └── ErrorBoundary.tsx
│       ├── store/
│       │   ├── useSessionStore.ts    # Zustand: sesión + datos
│       │   └── useChatStore.ts       # Zustand: historial chat
│       └── types/
│           └── index.ts             # TypeScript types
│
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml           # Django + React + Redis + Celery
│
├── supabase/
│   └── schema.sql                   # (mismo de v2)
│
└── .env.example
```

---

## 4. API REST — Endpoints

### Auth (`/api/auth/`)

```
POST   /api/auth/signup/          # Registro (email + password)
POST   /api/auth/login/           # Login → JWT access + refresh
POST   /api/auth/refresh/         # Refresh token
POST   /api/auth/logout/          # Invalidar token
GET    /api/auth/me/              # Perfil del usuario
```

### Search & Scraping (`/api/search/`)

```
POST   /api/search/               # Iniciar búsqueda → retorna job_id
GET    /api/search/{job_id}/      # Status del job (pending/running/done)
GET    /api/search/{job_id}/comments/   # Resultados paginados
```

### Dashboard (`/api/dashboard/`)

```
GET    /api/dashboard/{session_id}/sentiment/   # Datos para gráfico sentimiento
GET    /api/dashboard/{session_id}/emotions/    # Datos para gráfico emociones
GET    /api/dashboard/{session_id}/categories/  # Top categorías
GET    /api/dashboard/{session_id}/stats/       # Conteos por plataforma
```

### Chat RAG (`/api/chat/`)

```
POST   /api/chat/{session_id}/          # Pregunta → respuesta RAG
GET    /api/chat/{session_id}/history/  # Historial de conversación
```

### Export (`/api/export/`)

```
GET    /api/export/{session_id}/csv/    # Descargar CSV
GET    /api/export/{session_id}/xlsx/   # Descargar Excel
```

### Models (`/api/models/`)

```
GET    /api/models/                     # Listar modelos disponibles
GET    /api/models/providers/           # Status de providers (✅/❌)
```

---

## 5. Stack Tecnológico v3

| Capa | v2 (Gradio) | v3 (React + Django) |
|------|------------|---------------------|
| **Frontend** | Gradio Python | React 18 + Vite + TypeScript |
| **Estilos** | Gradio Themes | TailwindCSS + shadcn/ui |
| **Estado** | gr.State | Zustand |
| **Ruteo** | gr.Tabs | React Router v6 |
| **Gráficos** | Matplotlib/Plotly | Recharts / Plotly.js |
| **Backend** | Python monolítico | Django 5 + DRF |
| **Async tasks** | Generadores Gradio | Celery + Redis |
| **Auth** | No implementada | Supabase Auth JS (front) + JWT (back) |
| **API** | No tiene (monolito) | REST (DRF) + OpenAPI docs |
| **DB** | Supabase pgvector | Igual |
| **LLM** | google-genai + openai | Igual (desde backend) |
| **NLP** | PySentimiento + BART | Igual (desde backend) |
| **Deploy** | HF Spaces | Vercel (front) + Railway/Render (back) |
| **CI/CD** | Ninguno | GitHub Actions |

---

## 6. Plan de Ejecución (12 sprints de 1 semana)

### Sprint 1: Setup

- [x] Crear proyecto Django + DRF (`django-admin startproject`)
- [x] Crear proyecto React + Vite (`npm create vite@latest`)
- [x] Configurar Docker Compose (Django + React + Redis + PostgreSQL dev)
- [x] Configurar pre-commit hooks (black, ruff, eslint, prettier)
- [x] CI/CD: GitHub Actions para lint + test

### Sprint 2: Auth

- [x] Django: JWT auth con `djangorestframework-simplejwt`
- [x] Django: Supabase Auth integration (verificar tokens)
- [x] React: Login page + SignUp page
- [x] React: Supabase JS client → login/signup
- [x] React: Auth guard (redirect si no autenticado)
- [x] Axios interceptor (adjuntar JWT a requests)

### Sprint 3: Search UI + API

- [x] React: `SearchPage` con input + selector de redes sociales (iconos)
- [x] React: `ModelSelector` dropdown cargado desde API
- [x] React: `ProgressBar` en tiempo real (polling del job status)
- [x] Django: `POST /api/search/` → crea Celery task → retorna `job_id`
- [x] Django: `GET /api/search/{job_id}/` → status del job
- [x] Django: Migrar `tools/search_tool.py` → Django service

### Sprint 4: Scraping Services

- [x] Django: Migrar `tools/youtube_tool.py` → `scraping.services.youtube`
- [x] Django: Migrar `tools/reddit_tool.py` → `scraping.services.reddit`
- [x] Django: Migrar `tools/playstore_tool.py` → `scraping.services.playstore`
- [x] Django: Migrar `tools/apify_tool.py` → `scraping.services.apify`
- [x] Django: Unificar interfaz `BaseScraper` (abstract class)

### Sprint 5: Pipeline de Análisis

- [x] Django: Migrar `pipeline/orchestrator.py` → Celery task
- [x] Django: Migrar `pipeline/analyzer.py` → `analysis.services`
- [x] Django: Migrar `llm_manager.py` → `core/llm_manager.py`
- [x] Django: Migrar NLP (PySentimiento, BART) → `analysis.services`
- [x] Django: Migrar embeddings → `analysis.services.embeddings`
- [x] Celery: Encadenar tasks (scrape → analyze → store → embed)

### Sprint 6: Resultados + Dashboard

- [x] Django: `GET /api/search/{job_id}/comments/` paginado
- [x] Django: `GET /api/dashboard/{session_id}/...` endpoints
- [x] React: `DataTable` con columnas sortable
- [x] React: `DashboardCharts` con Recharts
- [x] React: Gráficos: sentimiento, emociones, categorías, conteo

### Sprint 7: Chat RAG

- [x] Django: Migrar `pipeline/rag.py` → `chat.services.rag`
- [x] Django: `POST /api/chat/{session_id}/` → respuesta RAG
- [x] Django: `GET /api/chat/{session_id}/history/`
- [x] React: `ChatPage` con burbujas de chat
- [x] React: Auto-scroll + loading states

### Sprint 8: Export + Models

- [x] Django: `GET /api/export/{session_id}/csv/`
- [x] Django: `GET /api/export/{session_id}/xlsx/`
- [x] Django: `GET /api/models/` con filtro de prohibidos + pricing
- [x] React: `ExportButtons` para CSV y Excel
- [x] React: Model selector con info de pricing

### Sprint 9: Historial + Sesiones

- [x] Django: Model `Session` + `ChatMessage`
- [x] Django: CRUD de sesiones por usuario
- [x] React: Sidebar con historial de sesiones
- [x] React: Click en sesión → carga datos previos

### Sprint 10: Polish + UX

- [x] React: Dark mode toggle
- [x] React: Responsive (mobile-first)
- [x] React: Skeleton loaders
- [x] React: Toast notifications (errores, éxito)
- [x] Django: Rate limiting (DRF throttling)
- [x] Django: OpenAPI schema (drf-spectacular)

### Sprint 11: Testing

- [x] Django: Unit tests para servicios
- [x] Django: Integration tests para API
- [x] React: Component tests (Vitest + Testing Library)
- [x] E2E: Cypress / Playwright (flujo completo)

### Sprint 12: Deploy

- [x] Frontend: Vercel (free tier) o S3 + CloudFront
- [x] Backend: Railway / Render / AWS ECS
- [x] Redis: Upstash (free tier) o ElastiCache
- [x] Supabase: Proyecto existente
- [x] CI/CD: Auto-deploy desde GitHub
- [x] Monitoring: Sentry (errors) + Grafana (métricas)

---

## 7. Riesgos y Mitigaciones

| Riesgo | Impacto | Mitigación |
|--------|---------|-----------|
| Supabase pgvector RPC no funciona igual desde Django | Alto | Testear `match_comments` RPC en sprint 1 |
| PySentimiento/BART carga lenta en Django | Medio | Lazy loading + cache de modelos en memoria |
| Celery + Redis añaden complejidad | Medio | Empezar con Django sync, añadir Celery en sprint 4 |
| React state management complejo | Medio | Zustand (simple). No usar Redux al inicio |
| Costo LLM se dispara con muchos usuarios | Medio | Rate limiting + cache de respuestas en Redis |

---

## 8. Costos v3 (producción)

| Servicio | USD/mes |
|----------|---------|
| Supabase Pro | $25 |
| SerpAPI | $50 |
| APIFY | $29 |
| Vercel (frontend) | $0 (hobby) |
| Railway/Render (backend) | $20-50 |
| Upstash Redis | $0 (free tier) |
| Gemini API | $0 (free tier) |
| **TOTAL** | **~$125-155** |

---

## 9. Timeline

```
S1  S2  S3  S4  S5  S6  S7  S8  S9  S10 S11 S12
▓   ▓   ▓   ▓   ▓   ▓   ▓   ▓   ▓   ▓   ▓   ▓
Setup   Auth  Search Scrap  Pipe  Dash  Chat Export Hist  UX   Test Deploy
│       │     │      │      │     │     │    │      │    │    │    │
└─MVP──┘     └──Core───────┘     └──Full────┘  └────Polish────┘
```

---

## 10. ¿Qué hacer con v2 mientras tanto?

v2 (Gradio) se mantiene como **fallback en producción** durante la migración. Ambas versiones comparten:

- Misma base de datos Supabase
- Mismas API keys
- Mismo schema `chismesito_gpt`

Cuando v3 esté estable (Sprint 8+), se hace el switch definitivo.
