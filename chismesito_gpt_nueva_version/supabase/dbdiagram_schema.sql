-- ╔══════════════════════════════════════════════════════════════════╗
-- ║        ChismesitoGPT v2 — Esquema para DBDiagram.io              ║
-- ║                                                                  ║
-- ║   IMPORTAR EN: https://dbdiagram.io                              ║
-- ║   Instrucciones: Haz clic en "Import" -> "Import from PostgreSQL"║
-- ║   y pega este contenido para generar el diagrama interactivo.    ║
-- ╚══════════════════════════════════════════════════════════════════╝

-- Tabla: allowed_users (Whitelist de control de acceso)
CREATE TABLE allowed_users (
    email VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT 'now()',
    notes TEXT
);

-- Tabla: user_requests (Tabla central de auditoría de búsquedas/sesiones)
CREATE TABLE user_requests (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    user_email VARCHAR NOT NULL,
    session_id UUID UNIQUE NOT NULL,
    query TEXT NOT NULL,
    social_medias VARCHAR NOT NULL,
    comments_count INTEGER DEFAULT 0,
    model VARCHAR,
    cost REAL DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT 'now()'
);

-- Tabla: request_performance (Métricas y profiling detallado)
CREATE TABLE request_performance (
    id UUID PRIMARY KEY,
    session_id UUID UNIQUE NOT NULL,
    execution_time REAL NOT NULL,
    llm_model VARCHAR NOT NULL,
    classification_time REAL NOT NULL,
    embedding_model VARCHAR NOT NULL,
    embedding_time REAL NOT NULL,
    max_comments INTEGER NOT NULL,
    num_social_medias INTEGER NOT NULL,
    social_medias TEXT[] NOT NULL,
    created_at TIMESTAMPTZ DEFAULT 'now()'
);

-- Tabla: unified_comments (Comentarios recolectados y su análisis)
CREATE TABLE unified_comments (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    session_id UUID NOT NULL,
    social_media VARCHAR NOT NULL,
    query TEXT NOT NULL,
    url TEXT,
    username VARCHAR,
    comment TEXT NOT NULL,
    rating REAL,
    likes INTEGER,
    post_date TIMESTAMPTZ,
    category VARCHAR,
    sentiment VARCHAR,
    emotion VARCHAR,
    embedding float8[], -- pgvector representado como array para compatibilidad
    extraction_date TIMESTAMPTZ DEFAULT 'now()',
    created_at TIMESTAMPTZ DEFAULT 'now()'
);

-- Tabla: chat_history (Historial del chatbot RAG)
CREATE TABLE chat_history (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    session_id UUID NOT NULL,
    role VARCHAR NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT 'now()'
);

-- ── RELACIONES (FOREIGN KEYS) ──

-- Vínculo entre auditoría de peticiones y whitelist de accesos
ALTER TABLE user_requests 
    ADD FOREIGN KEY (user_email) REFERENCES allowed_users(email) ON DELETE CASCADE;

-- Relación 1-a-1 entre profiling de rendimiento y la petición principal
ALTER TABLE request_performance 
    ADD FOREIGN KEY (session_id) REFERENCES user_requests(session_id) ON DELETE CASCADE;

-- Relación 1-a-Muchos entre la petición y los comentarios extraídos
ALTER TABLE unified_comments 
    ADD FOREIGN KEY (session_id) REFERENCES user_requests(session_id) ON DELETE CASCADE;

-- Relación 1-a-Muchos entre la petición y el historial de chat contextual
ALTER TABLE chat_history 
    ADD FOREIGN KEY (session_id) REFERENCES user_requests(session_id) ON DELETE CASCADE;
