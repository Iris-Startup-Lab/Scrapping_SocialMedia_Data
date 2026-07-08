# ChismesitoGPT v2 — Diagramas de Arquitectura

> **Actualizado a la versión nueva (`chismesito_gpt_nueva_version`).**
> La versión anterior (Shiny + Selenium/ChromeDriver + Pinecone, desplegada en AWS) es **obsoleta**.
> Esta guía documenta la arquitectura actual: **Gradio + LangChain Agent + APIFY + Supabase (pgvector)**, desplegada en **Hugging Face Spaces**.

---

## Arquitectura General del Sistema

```mermaid
flowchart TB
    %% ============ ESTILO DAILYDOSE OF DS ============
    classDef client    fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px,color:#1A1A1A;
    classDef app       fill:#D5E8D4,stroke:#82B366,stroke-width:2px,color:#1A1A1A;
    classDef search    fill:#FFF2CC,stroke:#D6B656,stroke-width:2px,color:#1A1A1A;
    classDef extract   fill:#FFE6CC,stroke:#D79B00,stroke-width:2px,color:#1A1A1A;
    classDef nlp       fill:#E1D5E7,stroke:#9673A6,stroke-width:2px,color:#1A1A1A;
    classDef storage   fill:#F8CECC,stroke:#B85450,stroke-width:2px,color:#1A1A1A;

    subgraph CL["👤 Capa Cliente"]
        BROWSER["Navegador Web<br/>Analista de Opinión"]
    end

    subgraph HF["🤗 Hugging Face Spaces · Docker + ZeroGPU"]
        direction TB
        UI["Gradio Wizard<br/>① Configuración · ② Dashboard · ③ Chat RAG"]
        AGENT["LangChain Agent<br/>Tool-calling · Gemini / DeepSeek / Claude"]
        UI --> AGENT
    end

    subgraph SRCH["🔎 Búsqueda de Enlaces Reales"]
        SERP["SerpAPI"]
        DDG["DuckDuckGo (respaldo)"]
    end

    subgraph EXT["🛰️ Extracción Multi-Plataforma"]
        YT["YouTube Data API v3"]
        RD["Reddit · PRAW"]
        PS["Google Play Store"]
        APIFY["APIFY Actors<br/>Facebook · IG · TikTok · X · Maps"]
    end

    subgraph ANL["🧠 Análisis NLP"]
        CAT["LLM → Categorías<br/>(muestra 20%)"]
        ZS["Zero-Shot · mDeBERTa / BART"]
        SENT["PySentimiento · RoBERTuito<br/>Sentimiento + Emoción"]
        EMB["Gemini Embeddings · 3072d"]
    end

    subgraph SB["(🗄️ Supabase · PostgreSQL + pgvector + Auth)"]
        direction TB
        UC["unified_comments<br/>relacional + vector"]
        CH["chat_history"]
        AU["allowed_users · whitelist"]
        RPC["match_comments RPC<br/>índice HNSW"]
        AUD["request_performance · user_requests"]
    end

    EXPORT["📤 Exportación CSV / Excel"]

    BROWSER -->|"HTTPS :7860"| UI
    AGENT -->|"buscar links"| SERP
    AGENT --> DDG
    AGENT -->|"API oficial"| YT
    AGENT --> RD
    AGENT --> PS
    AGENT -->|"APIFY"| APIFY
    AGENT --> CAT
    CAT --> ZS --> SENT --> EMB
    YT --> UC
    RD --> UC
    PS --> UC
    APIFY --> UC
    EMB -->|"insertar vectores"| UC
    AGENT -->|"login / RLS"| AU
    UC --> RPC
    RPC -->|"búsqueda semántica"| AGENT
    AGENT --> CH
    AGENT --> AUD
    UC -->|"dashboard Plotly"| UI
    UI --> EXPORT
    UC --> EXPORT

    class BROWSER client;
    class UI,AGENT app;
    class SERP,DDG search;
    class YT,RD,PS,APIFY extract;
    class CAT,ZS,SENT,EMB nlp;
    class UC,CH,AU,RPC,AUD storage;
```

---

## Flujo del Agente y Pipeline de Análisis

```mermaid
flowchart LR
    classDef in        fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px,color:#1A1A1A;
    classDef agent     fill:#D5E8D4,stroke:#82B366,stroke-width:2px,color:#1A1A1A;
    classDef tool      fill:#FFE6CC,stroke:#D79B00,stroke-width:2px,color:#1A1A1A;
    classDef store     fill:#F8CECC,stroke:#B85450,stroke-width:2px,color:#1A1A1A;
    classDef out       fill:#E1D5E7,stroke:#9673A6,stroke-width:2px,color:#1A1A1A;

    PROMPT["Prompt + Redes<br/>seleccionadas"] --> A
    subgraph A["LangChain Agent"]
        direction TB
        T1["search_web (SerpAPI)"]
        T2["get_*_comments (API/APIFY)"]
        T3["analyze_comments"]
        T4["store + vectorize"]
        T5["rag_chat"]
    end
    A -->|"links reales"| T2
    T2 -->|"DataFrame"| T3
    T3 -->|"enriquecido"| T4
    T4 --> DB[("Supabase<br/>pgvector")]
    DB -->|"match_comments"| T5
    T5 --> CHAT["Chat RAG<br/>respuestas"]

    class PROMPT in;
    class A agent;
    class T1,T2,T3,T4,T5 tool;
    class DB store;
    class CHAT out;
```

---

## Opción 1: CLI de nanobanana

```bash
nanobanana architecture "Cloud architecture for ChismesitoGPT v2, an AI social-listening platform. The system is deployed on Hugging Face Spaces as a Docker container with ZeroGPU. A Gradio Wizard UI (3 steps: Configuration, Dashboard, RAG Chat) runs a LangChain tool-calling agent that orchestrates LLMs (Gemini, DeepSeek, Claude). The agent uses SerpAPI and DuckDuckGo to find real social-media links, then extracts comments via official APIs (YouTube Data API v3, Reddit PRAW, Google Play Store) and APIFY actors (Facebook, Instagram, TikTok, X/Twitter, Google Maps). An NLP pipeline classifies topics (LLM zero-shot), sentiment and emotion (PySentimiento / RoBERTuito), and generates 3072-d Gemini embeddings. All data lands in Supabase PostgreSQL with pgvector (HNSW index, match_comments RPC), Supabase Auth + whitelist, and audit tables. Plotly dashboards and CSV/Excel exports complete the flow. Show data flows and color-code by layer: client, app, search, extraction, NLP, storage. The RAG works with DeepSeek or Gemini or ChatGPT or Claude and in the future will work with Mistral or GLM"
```

## Opción 2: Diagrama técnico detallado con /diagram

```
/diagram Architecture for ChismesitoGPT v2 social-media analytics app deployed on Hugging Face Spaces (Docker + ZeroGPU): Gradio Wizard UI (Config, Dashboard, RAG Chat) -> LangChain tool-calling Agent (Gemini/DeepSeek/Claude) -> SerpAPI/DuckDuckGo search -> official APIs (YouTube, Reddit, Play Store) and APIFY (FB, IG, TikTok, X, Maps). NLP pipeline: LLM categories + zero-shot + PySentimiento + Gemini 3072-d embeddings -> Supabase PostgreSQL + pgvector (HNSW, match_comments RPC, Auth whitelist, audit logs). Plotly dashboards, CSV/Excel export. --type=architecture --complexity=comprehensive --style=technical
```

## Opción 3: Prompt para MCP Nano Banana

```
Generate a clean, professional cloud architecture diagram for ChismesitoGPT v2, an AI social-listening and opinion-mining platform.

The application runs on Hugging Face Spaces as a Docker container with ZeroGPU acceleration. A Gradio Wizard UI (3 steps: Configuration, Dashboard, RAG Chat) drives a LangChain tool-calling agent that orchestrates multiple LLMs (Google Gemini, DeepSeek, Claude).

Architecture components:
1. Client: Web browser (analyst) accessing the Gradio UI over HTTPS on port 7860.
2. App layer (Hugging Face Spaces): Gradio Wizard + LangChain Agent.
3. Link search: SerpAPI (primary) and DuckDuckGo (fallback) to discover real social-media URLs.
4. Extraction: Official APIs — YouTube Data API v3, Reddit (PRAW), Google Play Store; and APIFY actors for Facebook, Instagram, TikTok, X/Twitter, and Google Maps.
5. NLP / analysis: LLM-generated topic categories (from a 20% sample), zero-shot classifier (mDeBERTa / BART), sentiment & emotion via PySentimiento (RoBERTuito), and 3072-d Gemini embeddings.
6. Storage: Supabase PostgreSQL with pgvector — unified_comments (relational + vector), chat_history, allowed_users (whitelist), match_comments RPC with HNSW index, and audit tables (request_performance, user_requests). Supabase Auth handles login and Row Level Security.
7. Output: Plotly dashboards and CSV/Excel export.
8. RAG works with DeepSeek or Gemini or ChatGPT or Claude and in the future will work with Mistral or GLM

Style: Corporate, modern diagram in the visual language of "Daily Dose of Data Science" — flat rounded nodes, soft pastel color palette grouping components by layer (client = light blue, app = light green, search = light yellow, extraction = light orange, NLP = light purple, storage = light red/pink), thin consistent strokes, clear labeled arrows showing data-flow direction. No 3D, no heavy gradients.
```
