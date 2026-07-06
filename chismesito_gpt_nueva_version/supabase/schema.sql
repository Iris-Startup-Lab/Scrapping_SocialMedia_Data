-- ╔══════════════════════════════════════════════════════════════════╗
-- ║      ChismesitoGPT v2 — Schema: chismesito_gpt (explícito)      ║
-- ║      Ejecutar en Supabase SQL Editor                             ║
-- ║                                                                  ║
-- ║  PRIMERO: Dashboard → Database → Extensions                     ║
-- ║  Activa: "vector" y "uuid-ossp" (2 clics)                       ║
-- ║                                                                  ║
-- ║  NOTA: Las extensiones (vector, uuid-ossp) siempre van en        ║
-- ║  el schema "extensions" o "public" (limitación de Supabase).    ║
-- ║  Todas las TABLAS y FUNCIONES de la app van en chismesito_gpt.  ║
-- ╚══════════════════════════════════════════════════════════════════╝

-- ── Extensiones (Supabase solo permite instalarlas en public/extensions) ──
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA extensions;
CREATE EXTENSION IF NOT EXISTS "vector"    WITH SCHEMA extensions;

-- ── Crear esquema de la aplicación ───────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS chismesito_gpt;

-- ─────────────────────────────────────────────────────────────────────────
-- TABLA: unified_comments
-- Comentarios consolidados de todas las redes sociales
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chismesito_gpt.unified_comments (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID        NOT NULL,
    session_id      UUID        NOT NULL DEFAULT gen_random_uuid(),
    social_media    TEXT        NOT NULL CHECK (social_media IN (
                                    'youtube','x_twitter','reddit','facebook',
                                    'instagram','tiktok','google_maps','playstore'
                                )),
    query           TEXT        NOT NULL,
    url             TEXT,
    username        TEXT,
    comment         TEXT        NOT NULL,
    rating          REAL,
    likes           INTEGER,
    post_date       TIMESTAMPTZ,
    category        TEXT,
    sentiment       TEXT        CHECK (sentiment IN ('Positivo','Negativo','Neutral','Error')),
    emotion         TEXT        CHECK (emotion IN (
                                    'Alegria','Tristeza','Enojo','Miedo',
                                    'Sorpresa','Asco','Neutral','Error'
                                )),
    embedding       VECTOR(3072),   -- gemini-embedding-2 produce 3072 dimensiones
    extraction_date TIMESTAMPTZ DEFAULT NOW(),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────
-- TABLA: chat_history
-- Historial de mensajes del chat RAG
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chismesito_gpt.chat_history (
    id          UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID    NOT NULL,
    session_id  UUID    NOT NULL,
    role        TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT    NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────
-- TABLA: user_requests
-- Registro de cada búsqueda / petición de usuario
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chismesito_gpt.user_requests (
    id              UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID    NOT NULL,
    user_email      TEXT    NOT NULL,
    session_id      UUID    NOT NULL UNIQUE,
    query           TEXT    NOT NULL,
    social_medias   TEXT    NOT NULL,   -- lista serializada, ej: "youtube,reddit"
    comments_count  INTEGER DEFAULT 0,
    model           TEXT,
    cost            REAL    DEFAULT 0.0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────
-- TABLA: request_performance
-- Métricas de ejecución por petición (profiling)
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chismesito_gpt.request_performance (
    id                  UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID    NOT NULL UNIQUE
                                REFERENCES chismesito_gpt.user_requests(session_id)
                                ON DELETE CASCADE,
    execution_time      REAL    NOT NULL,   -- segundos totales
    llm_model           TEXT    NOT NULL,
    classification_time REAL    NOT NULL,   -- segundos de BART zero-shot
    embedding_model     TEXT    NOT NULL,
    embedding_time      REAL    NOT NULL,   -- segundos de Gemini embeddings
    max_comments        INTEGER NOT NULL,
    num_social_medias   INTEGER NOT NULL,
    social_medias       TEXT[]  NOT NULL,   -- ej: ARRAY['youtube','reddit']
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────
-- FOREIGN KEYS: unified_comments y chat_history → user_requests
-- Garantizan integridad referencial entre sesiones
-- ─────────────────────────────────────────────────────────────────────────
ALTER TABLE chismesito_gpt.unified_comments
    DROP CONSTRAINT IF EXISTS fk_comments_session_id,
    ADD  CONSTRAINT fk_comments_session_id
    FOREIGN KEY (session_id)
    REFERENCES chismesito_gpt.user_requests(session_id)
    ON DELETE CASCADE;

ALTER TABLE chismesito_gpt.chat_history
    DROP CONSTRAINT IF EXISTS fk_chat_session_id,
    ADD  CONSTRAINT fk_chat_session_id
    FOREIGN KEY (session_id)
    REFERENCES chismesito_gpt.user_requests(session_id)
    ON DELETE CASCADE;

-- ─────────────────────────────────────────────────────────────────────────
-- ÍNDICES — todos con prefijo explícito chismesito_gpt.
-- ─────────────────────────────────────────────────────────────────────────

-- unified_comments
CREATE INDEX IF NOT EXISTS idx_uc_session
    ON chismesito_gpt.unified_comments (session_id);
CREATE INDEX IF NOT EXISTS idx_uc_user
    ON chismesito_gpt.unified_comments (user_id);
CREATE INDEX IF NOT EXISTS idx_uc_media
    ON chismesito_gpt.unified_comments (social_media);
CREATE INDEX IF NOT EXISTS idx_uc_sentiment
    ON chismesito_gpt.unified_comments (sentiment);
CREATE INDEX IF NOT EXISTS idx_uc_category
    ON chismesito_gpt.unified_comments (category);
CREATE INDEX IF NOT EXISTS idx_uc_created_at
    ON chismesito_gpt.unified_comments (created_at DESC);

-- Indice semantico: hnsw con cast a halfvec (soporta hasta 4000d, pgvector 0.7+)
-- ivfflat tiene limite de 2000d y no es compatible con VECTOR(3072)
CREATE INDEX IF NOT EXISTS idx_uc_embedding
    ON chismesito_gpt.unified_comments
    USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops);

-- chat_history
CREATE INDEX IF NOT EXISTS idx_chat_session
    ON chismesito_gpt.chat_history (session_id);
CREATE INDEX IF NOT EXISTS idx_chat_user
    ON chismesito_gpt.chat_history (user_id);
CREATE INDEX IF NOT EXISTS idx_chat_created_at
    ON chismesito_gpt.chat_history (created_at DESC);

-- user_requests
CREATE INDEX IF NOT EXISTS idx_ur_session
    ON chismesito_gpt.user_requests (session_id);
CREATE INDEX IF NOT EXISTS idx_ur_user
    ON chismesito_gpt.user_requests (user_id);
CREATE INDEX IF NOT EXISTS idx_ur_created_at
    ON chismesito_gpt.user_requests (created_at DESC);

-- request_performance
CREATE INDEX IF NOT EXISTS idx_rp_session
    ON chismesito_gpt.request_performance (session_id);
CREATE INDEX IF NOT EXISTS idx_rp_created_at
    ON chismesito_gpt.request_performance (created_at DESC);

-- ─────────────────────────────────────────────────────────────────────────
-- FUNCIÓN RPC: match_comments
-- Búsqueda semántica por similitud coseno (pgvector)
-- Llamada desde Python: supabase.rpc("match_comments", {...})
-- ─────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION chismesito_gpt.match_comments(
    query_embedding VECTOR(3072),
    p_session_id    UUID,
    p_user_id       UUID,
    match_count     INT DEFAULT 10
)
RETURNS TABLE (
    id           UUID,
    comment      TEXT,
    social_media TEXT,
    sentiment    TEXT,
    emotion      TEXT,
    category     TEXT,
    similarity   FLOAT
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
        -- Cast a halfvec para usar el indice hnsw (evita sequential scan)
        1 - (uc.embedding::halfvec(3072) <=> query_embedding::halfvec(3072)) AS similarity
    FROM chismesito_gpt.unified_comments uc
    WHERE uc.session_id = p_session_id
      AND uc.user_id    = p_user_id
      AND uc.embedding  IS NOT NULL
    ORDER BY uc.embedding::halfvec(3072) <=> query_embedding::halfvec(3072)
    LIMIT match_count;
END;
$$;

-- ─────────────────────────────────────────────────────────────────────────
-- RLS — Deshabilitado para desarrollo local
-- En producción: ejecutar supabase/rls_policies.sql
-- ─────────────────────────────────────────────────────────────────────────
ALTER TABLE chismesito_gpt.unified_comments    DISABLE ROW LEVEL SECURITY;
ALTER TABLE chismesito_gpt.chat_history        DISABLE ROW LEVEL SECURITY;
ALTER TABLE chismesito_gpt.user_requests       DISABLE ROW LEVEL SECURITY;
ALTER TABLE chismesito_gpt.request_performance DISABLE ROW LEVEL SECURITY;