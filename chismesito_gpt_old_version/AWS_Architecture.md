# Iris Social Media Downloader - Arquitectura AWS

## Servicios Principales Detectados en la App

| Capa | Servicio/Componente | Tecnología Actual |
|------|-------------------|-------------------|
| **Frontend + Backend** | Shiny Python (app.py, ui.py, server.py) | Python Shiny |
| **Browser Automation** | Selenium + ChromeDriver | selenium, playwright |
| **Web Scraping** | APIs + Selenium (Twitter, YouTube, FB, IG, Reddit, Maps) | tweepy, google-api-client, praw, RapidAPI |
| **LLMs** | Gemini, OpenAI, OpenRouter, DeepSeek | google-genai, openai, langchain |
| **NLP** | Sentiment, Emotions, Topics, Summarization | spaCy, transformers, pysentimiento |
| **Base de Datos** | PostgreSQL vía Supabase | psycopg2, SQLAlchemy, supabase-py |
| **Vector DB** | Pinecone (cosine similarity, 768d) | pinecone-client |
| **Visualización** | Plotly, Matplotlib, PyVis, Folium | plotly, matplotlib, pyvis |
| **Exportación** | Excel, PDF, CSV | openpyxl, weasyprint |
| **Caché** | Directorios locales en disco | Sistema de archivos |

## Servicios AWS Necesarios para Ejecución en AWS

### 1. Cómputo (Shiny App)
- **AWS ECS con Fargate** (serverless containers)
  - Contenedor Docker con Python + ChromeDriver + Playwright
  - Escalamiento automático basado en carga
  - Mínimo 2 GB RAM, 1 vCPU (recomendado 4 GB / 2 vCPU por la carga de Selenium + LLMs)

### 2. Balanceo de Carga
- **Application Load Balancer (ALB)**
  - Routing HTTPS hacia ECS
  - WebSockets necesarios para Shiny (ALB soporta WebSockets)
  - Health checks contra la app Shiny

### 3. Base de Datos Relacional
- **Amazon RDS for PostgreSQL**
  - Reemplazo directo de Supabase
  - Versión 15+ (compatible con schemas existentes)
  - db.t3.medium (mínimo) o Aurora Serverless v2
  - Almacenamiento: 20-100 GB gp3

### 4. Almacenamiento de Objetos
- **Amazon S3**
  - Assets estáticos (logos, CSS, JS)
  - Archivos de exportación (Excel, CSV, PDF)
  - Datos demo y templates
  - Logs de scraping

### 5. Vector Database
- **Opción A: Amazon OpenSearch Serverless** (vector engine)
  - Compatible con cosine similarity
  - Indexación de embeddings (768d Gemini)
- **Opción B: Mantener Pinecone** (más sencillo, sin migración)
- **Opción C: pgvector en RDS** (simplifica stack, mismo PostgreSQL)

### 6. Contenedores
- **Amazon ECR (Elastic Container Registry)**
  - Almacenamiento de imágenes Docker
  - Integración nativa con ECS

### 7. Secretos y Configuración
- **AWS Secrets Manager**
  - API Keys (Gemini, OpenAI, Twitter, YouTube, Reddit, etc.)
  - Credenciales de base de datos
  - Reemplazo del archivo `.env`

### 8. Red y DNS
- **Amazon Route 53** - DNS y dominio
- **AWS Certificate Manager (ACM)** - SSL/TLS gratuito
- **Amazon CloudFront** - CDN para assets estáticos (opcional)

### 9. Monitoreo
- **Amazon CloudWatch**
  - Logs de la aplicación (streaming desde ECS)
  - Métricas de CPU/Memoria
  - Alarmas de rendimiento
  - Dashboard de operación

### 10. Seguridad
- **AWS WAF** - Web Application Firewall (protección contra bots, SQLi, XSS)
- **AWS Shield** - Protección DDoS (incluido con ALB)
- **AWS IAM** - Roles y políticas de acceso

### 11. Opcionales (Recomendados para Producción)

| Servicio | Propósito |
|----------|-----------|
| **Amazon SQS** | Cola de trabajos de scraping asíncronos |
| **AWS Step Functions** | Orquestación de scraping multi-plataforma |
| **AWS Lambda** | Post-procesamiento ligero (generar PDF, limpiar datos) |
| **Amazon ElastiCache (Redis)** | Caché de sesiones Shiny y rate limiting |

## Diagrama de Arquitectura

```
                        ┌─────────────┐
                        │  Route 53   │
                        │    DNS       │
                        └──────┬──────┘
                               │
                        ┌──────┴──────┐
                        │  CloudFront  │
                        │   + WAF     │
                        └──────┬──────┘
                               │
                        ┌──────┴──────┐
                        │     ALB     │
                        │  HTTPS/WS   │
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
       ┌──────┴──────┐  ┌─────┴──────┐  ┌──────┴──────┐
       │  ECS Task   │  │  ECS Task  │  │  ECS Task   │
       │  (Shiny)    │  │  (Shiny)   │  │  (Shiny)    │
       │ Chrome + Py │  │ Chrome + Py│  │ Chrome + Py │
       └──────┬──────┘  └─────┬──────┘  └──────┬──────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
       ┌──────┴──────┐  ┌─────┴──────┐  ┌──────┴──────┐
       │   RDS PG    │  │OpenSearch  │  │     S3      │
       │ PostgreSQL  │  │Vector DB   │  │  Assets     │
       └─────────────┘  └────────────┘  └─────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        ┌─────┴─────┐   ┌─────┴─────┐   ┌──────┴──────┐
        │ External  │   │  Secrets  │   │  CloudWatch │
        │ APIs      │   │  Manager  │   │   Logs      │
        │(Google,   │   └───────────┘   └─────────────┘
        │ Twitter,  │
        │ OpenAI,   │
        │ etc.)     │
        └───────────┘
```

## Flujo de Datos

1. Usuario accede vía Route 53 → CloudFront → ALB
2. ALB distribuye tráfico a ECS Fargate (Shiny App)
3. Shiny App se conecta a RDS PostgreSQL para datos de usuario y scraping
4. Shiny usa OpenSearch Serverless (o Pinecone) para búsqueda semántica
5. Assets estáticos y exportaciones se almacenan en S3
6. LLMs y APIs externas se consumen directamente desde ECS (no requieren servicio AWS)
7. Secrets Manager provee API keys de forma segura
8. CloudWatch centraliza logs y métricas

## Consideraciones

- **Selenium/Chrome**: Ejecutar dentro del contenedor ECS con `--headless`. Se puede usar `selenium-wire` con `undetected-chromedriver`. Requiere ~500 MB extra por tarea.
- **Cold Starts**: Shiny App tiene carga inicial pesada (modelos NLP + conexiones). Usar ECS con `warm pool` o mínimo 2 tareas siempre activas.
- **Costos**: Aprox. estimado:
  - ECS Fargate: ~$30-60/mes (2 tareas 2GB/1vCPU)
  - RDS PostgreSQL: ~$20-40/mes (db.t3.medium)
  - ALB: ~$20/mes
  - S3: ~$1-5/mes
  - Secrets Manager: ~$5/mes
  - **Total estimado: ~$80-130/mes**
