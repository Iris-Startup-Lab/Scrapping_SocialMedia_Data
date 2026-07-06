-- ══════════════════════════════════════════════════════════════════════
-- ChismesitoGPT v2 — RLS Policies + Whitelist Table
-- Ejecutar en: Supabase SQL Editor
-- Proyecto: COMPARTIDO (NO tocar Auth Settings globales)
-- Schema: chismesito_gpt
-- ══════════════════════════════════════════════════════════════════════
--
-- INSTRUCCIONES:
--   1. Abre Supabase Dashboard → SQL Editor → New Query
--   2. Pega TODO este archivo y haz clic en "Run"
--   3. Verifica que no haya errores en la consola inferior
--
-- ══════════════════════════════════════════════════════════════════════

-- ──────────────────────────────────────────────────────────────────────
-- PASO 1: Habilitar RLS en tablas existentes
-- (SIN Foreign Keys a auth.users — proyecto compartido, no arriesgamos)
-- ──────────────────────────────────────────────────────────────────────

ALTER TABLE chismesito_gpt.unified_comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE chismesito_gpt.chat_history ENABLE ROW LEVEL SECURITY;

-- ──────────────────────────────────────────────────────────────────────
-- PASO 2: Policies granulares para unified_comments
-- SELECT e INSERT por separado (no FOR ALL — más control)
-- ──────────────────────────────────────────────────────────────────────

-- Borrar si ya existían (para poder re-ejecutar el script)
DROP POLICY IF EXISTS "user_select_own_comments" ON chismesito_gpt.unified_comments;
DROP POLICY IF EXISTS "user_insert_own_comments" ON chismesito_gpt.unified_comments;

-- Solo leer comentarios propios
CREATE POLICY "user_select_own_comments"
    ON chismesito_gpt.unified_comments
    FOR SELECT
    USING (auth.uid() = user_id);

-- Solo insertar con user_id propio
CREATE POLICY "user_insert_own_comments"
    ON chismesito_gpt.unified_comments
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- ──────────────────────────────────────────────────────────────────────
-- PASO 3: Policies para chat_history
-- ──────────────────────────────────────────────────────────────────────

DROP POLICY IF EXISTS "user_select_own_chat" ON chismesito_gpt.chat_history;
DROP POLICY IF EXISTS "user_insert_own_chat" ON chismesito_gpt.chat_history;

-- Solo leer historial de chat propio
CREATE POLICY "user_select_own_chat"
    ON chismesito_gpt.chat_history
    FOR SELECT
    USING (auth.uid() = user_id);

-- Solo insertar con user_id propio
CREATE POLICY "user_insert_own_chat"
    ON chismesito_gpt.chat_history
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- ──────────────────────────────────────────────────────────────────────
-- PASO 4: Tabla allowed_users (whitelist de usuarios autorizados)
-- Esta tabla es el control de acceso de ChismesitoGPT.
-- La verifica el backend con service_key ANTES de procesar el login.
-- ──────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS chismesito_gpt.allowed_users (
    email       TEXT PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT 'Usuario',
    active      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    notes       TEXT  -- campo libre para notas del admin
);

COMMENT ON TABLE chismesito_gpt.allowed_users IS
    'Whitelist de usuarios autorizados para ChismesitoGPT. Gestionada por el admin.';

COMMENT ON COLUMN chismesito_gpt.allowed_users.active IS
    'FALSE = usuario bloqueado. Efecto inmediato en el siguiente intento de login.';

-- ──────────────────────────────────────────────────────────────────────
-- PASO 5: Proteger allowed_users del acceso público
-- Solo el backend (service_key) puede leer/escribir esta tabla.
-- La anon_key (login del browser) NO puede acceder.
-- ──────────────────────────────────────────────────────────────────────

ALTER TABLE chismesito_gpt.allowed_users ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "deny_anon_access" ON chismesito_gpt.allowed_users;

CREATE POLICY "deny_anon_access"
    ON chismesito_gpt.allowed_users
    FOR ALL TO anon
    USING (false);

-- ──────────────────────────────────────────────────────────────────────
-- PASO 6: Insertar usuarios autorizados
-- ⚠️  EDITA ESTOS VALORES antes de ejecutar
-- ──────────────────────────────────────────────────────────────────────

-- Borrar e insertar para poder re-ejecutar sin duplicados
INSERT INTO chismesito_gpt.allowed_users (email, name, notes)
VALUES
    ('tu@email.com', 'Admin', 'Cuenta principal del administrador')
ON CONFLICT (email) DO NOTHING;

-- Email de sistema para peticiones anónimas / pruebas locales sin sesión activa
INSERT INTO chismesito_gpt.allowed_users (email, name, active, notes)
VALUES
    ('anonimo@chismesitogpt.com', 'Sistema (Anónimo)', TRUE,
     'Usuario de sistema. Usado por el pipeline cuando no hay sesión de usuario activa (pruebas, API calls).')
ON CONFLICT (email) DO NOTHING;

-- Para agregar más usuarios en el futuro, ejecuta:
-- INSERT INTO chismesito_gpt.allowed_users (email, name)
-- VALUES ('nuevo@email.com', 'Nombre del usuario');


-- ──────────────────────────────────────────────────────────────────────
-- PASO 7: RLS y políticas para user_requests y request_performance
-- ──────────────────────────────────────────────────────────────────────

-- Habilitar RLS
ALTER TABLE chismesito_gpt.user_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE chismesito_gpt.request_performance ENABLE ROW LEVEL SECURITY;

-- Limpiar políticas anteriores si existían
DROP POLICY IF EXISTS "user_select_own_requests" ON chismesito_gpt.user_requests;
DROP POLICY IF EXISTS "user_insert_own_requests" ON chismesito_gpt.user_requests;
DROP POLICY IF EXISTS "user_select_own_performance" ON chismesito_gpt.request_performance;
DROP POLICY IF EXISTS "user_insert_own_performance" ON chismesito_gpt.request_performance;

-- Políticas para user_requests
CREATE POLICY "user_select_own_requests"
    ON chismesito_gpt.user_requests
    FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "user_insert_own_requests"
    ON chismesito_gpt.user_requests
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Políticas para request_performance
CREATE POLICY "user_select_own_performance"
    ON chismesito_gpt.request_performance
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM chismesito_gpt.user_requests ur
            WHERE ur.session_id = request_performance.session_id
              AND ur.user_id = auth.uid()
        )
    );

CREATE POLICY "user_insert_own_performance"
    ON chismesito_gpt.request_performance
    FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM chismesito_gpt.user_requests ur
            WHERE ur.session_id = request_performance.session_id
              AND ur.user_id = auth.uid()
        )
    );

-- ──────────────────────────────────────────────────────────────────────
-- VERIFICACIÓN (ejecuta estas queries por separado para confirmar)
-- ──────────────────────────────────────────────────────────────────────

-- SELECT tablename, rowsecurity FROM pg_tables
-- WHERE schemaname = 'chismesito_gpt';
-- ↑ Debe mostrar rowsecurity = true para unified_comments, chat_history, allowed_users

-- SELECT * FROM chismesito_gpt.allowed_users;
-- ↑ Debe mostrar los usuarios que insertaste

-- SELECT schemaname, tablename, policyname, cmd, qual
-- FROM pg_policies
-- WHERE schemaname = 'chismesito_gpt';
-- ↑ Debe mostrar las 5 policies creadas
