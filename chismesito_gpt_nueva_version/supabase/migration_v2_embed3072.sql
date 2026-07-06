-- ╔══════════════════════════════════════════════════════════════════╗
-- ║   ChismesitoGPT v2 — MIGRACIÓN SEGURA (datos existentes)        ║
-- ║   Ejecutar en: Supabase SQL Editor                               ║
-- ║                                                                  ║
-- ║   SEGURO: No borra tablas ni comentarios.                        ║
-- ║   Solo actualiza columna embedding, función RPC,                 ║
-- ║   crea tablas nuevas y agrega índices faltantes.                 ║
-- ╚══════════════════════════════════════════════════════════════════╝

-- ─────────────────────────────────────────────────────────────────────────
-- BLOQUE 1: Asegurar que el schema existe
-- ─────────────────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS chismesito_gpt;


-- ─────────────────────────────────────────────────────────────────────────
-- BLOQUE 2: Migrar columna embedding de 768d -> 3072d
-- Los embeddings existentes se pierden (son regenerables)
-- Los comentarios, sentimientos, categorias, etc. se conservan
-- ─────────────────────────────────────────────────────────────────────────

-- Eliminar indice ivfflat primero (no se puede alterar tipo con indice activo)
DROP INDEX IF EXISTS chismesito_gpt.idx_uc_embedding;
DROP INDEX IF EXISTS chismesito_gpt.idx_unified_embedding;  -- nombre anterior si existia

-- Cambiar la columna: drop + add (pgvector no permite ALTER COLUMN TYPE para VECTOR)
ALTER TABLE chismesito_gpt.unified_comments DROP COLUMN IF EXISTS embedding;
ALTER TABLE chismesito_gpt.unified_comments ADD  COLUMN embedding VECTOR(3072);

-- hnsw con cast a halfvec: soporta hasta 4000d (pgvector 0.7+)
-- ivfflat tiene limite de 2000d y no funciona con 3072d
CREATE INDEX idx_uc_embedding
    ON chismesito_gpt.unified_comments
    USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops);


-- ─────────────────────────────────────────────────────────────────────────
-- BLOQUE 3: Actualizar funcion RPC match_comments a 3072d
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
-- BLOQUE 4: Crear tablas nuevas (IF NOT EXISTS = no toca las existentes)
-- ─────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS chismesito_gpt.user_requests (
    id              UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID    NOT NULL,
    user_email      TEXT    NOT NULL,
    session_id      UUID    NOT NULL UNIQUE,
    query           TEXT    NOT NULL,
    social_medias   TEXT    NOT NULL,
    comments_count  INTEGER DEFAULT 0,
    model           TEXT,
    cost            REAL    DEFAULT 0.0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chismesito_gpt.request_performance (
    id                  UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID    NOT NULL UNIQUE
                                REFERENCES chismesito_gpt.user_requests(session_id)
                                ON DELETE CASCADE,
    execution_time      REAL    NOT NULL,
    llm_model           TEXT    NOT NULL,
    classification_time REAL    NOT NULL,
    embedding_model     TEXT    NOT NULL,
    embedding_time      REAL    NOT NULL,
    max_comments        INTEGER NOT NULL,
    num_social_medias   INTEGER NOT NULL,
    social_medias       TEXT[]  NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);


-- ─────────────────────────────────────────────────────────────────────────
-- BLOQUE 5: FK de unified_comments y chat_history -> user_requests
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
-- BLOQUE 6: Indices (IF NOT EXISTS = no falla si ya existen)
-- ─────────────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_uc_session    ON chismesito_gpt.unified_comments (session_id);
CREATE INDEX IF NOT EXISTS idx_uc_user       ON chismesito_gpt.unified_comments (user_id);
CREATE INDEX IF NOT EXISTS idx_uc_media      ON chismesito_gpt.unified_comments (social_media);
CREATE INDEX IF NOT EXISTS idx_uc_sentiment  ON chismesito_gpt.unified_comments (sentiment);
CREATE INDEX IF NOT EXISTS idx_uc_category   ON chismesito_gpt.unified_comments (category);
CREATE INDEX IF NOT EXISTS idx_uc_created_at ON chismesito_gpt.unified_comments (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_chat_session    ON chismesito_gpt.chat_history (session_id);
CREATE INDEX IF NOT EXISTS idx_chat_user       ON chismesito_gpt.chat_history (user_id);
CREATE INDEX IF NOT EXISTS idx_chat_created_at ON chismesito_gpt.chat_history (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ur_session    ON chismesito_gpt.user_requests (session_id);
CREATE INDEX IF NOT EXISTS idx_ur_user       ON chismesito_gpt.user_requests (user_id);
CREATE INDEX IF NOT EXISTS idx_ur_created_at ON chismesito_gpt.user_requests (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_rp_session    ON chismesito_gpt.request_performance (session_id);
CREATE INDEX IF NOT EXISTS idx_rp_created_at ON chismesito_gpt.request_performance (created_at DESC);


-- ─────────────────────────────────────────────────────────────────────────
-- BLOQUE 7: Verificacion final (ejecuta por separado)
-- ─────────────────────────────────────────────────────────────────────────

-- Verifica que embedding es ahora VECTOR(3072):
-- SELECT column_name, udt_name
-- FROM information_schema.columns
-- WHERE table_schema = 'chismesito_gpt'
--   AND table_name = 'unified_comments'
--   AND column_name = 'embedding';

-- Verifica que tus datos siguen intactos:
-- SELECT COUNT(*) FROM chismesito_gpt.unified_comments;

-- Verifica que la funcion RPC existe:
-- SELECT routine_name FROM information_schema.routines
-- WHERE routine_schema = 'chismesito_gpt';
