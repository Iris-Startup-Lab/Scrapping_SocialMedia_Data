# Plan de Corrección: Integración de `user_requests` y Ecosistema de Base de Datos

Este documento detalla los cambios exactos requeridos para solucionar los errores críticos de esquemas, dimensiones de embeddings y restricciones de claves foráneas detectados en la tabla de logs de peticiones y rendimiento.

---

## 1. Ajustes en la Base de Datos (`supabase/schema.sql` y `supabase/rls_policies.sql`)

### A. schema.sql
Modificar el archivo `supabase/schema.sql` para:
1. Crear el esquema `chismesito_gpt` e incluirlo en el `search_path` al inicio.
2. Cambiar la dimensión del vector de embedding de `768` a `3072` (para soportar `gemini-embedding-2`).
3. Eliminar la declaración redundante de índices.

#### Cambios propuestos:

```diff
+ -- Crear y establecer esquema de trabajo
+ CREATE SCHEMA IF NOT EXISTS chismesito_gpt;
+ SET search_path TO chismesito_gpt, public;

-- Tabla unificada de comentarios
CREATE TABLE IF NOT EXISTS unified_comments (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL,
    session_id      UUID NOT NULL DEFAULT gen_random_uuid(),
    social_media    TEXT NOT NULL CHECK (social_media IN (
        'youtube','x_twitter','reddit','facebook','instagram',
        'tiktok','google_maps','playstore'
    )),
    query           TEXT NOT NULL,
    url             TEXT,
    username        TEXT,
    comment         TEXT NOT NULL,
    rating          REAL,
    likes           INTEGER,
    post_date       TIMESTAMPTZ,
    category        TEXT,
    sentiment       TEXT CHECK (sentiment IN ('Positivo','Negativo','Neutral','Error')),
    emotion         TEXT CHECK (emotion IN (
        'Alegria','Tristeza','Enojo','Miedo','Sorpresa','Asco','Neutral','Error'
    )),
-   embedding       VECTOR(768),
+   embedding       VECTOR(3072),
    extraction_date TIMESTAMPTZ DEFAULT NOW(),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Funcion RPC: match_comments
CREATE OR REPLACE FUNCTION match_comments(
-   query_embedding VECTOR(768),
+   query_embedding VECTOR(3072),
    p_session_id UUID,
    p_user_id UUID,
    match_count INT DEFAULT 10
)
```

> [!NOTE]
> Se deben eliminar las líneas 35-40 (índices duplicados sin prefijo de esquema) y conservar únicamente el bloque de índices finales de las líneas 142-155.

---

## 2. Ajuste en la Lógica de Negocio de Python (`pipeline/orchestrator.py`)

Para evitar la violación de clave foránea (`fk_comments_session_id`) cuando `user_email` es nulo o vacío (por ejemplo, en ejecuciones automáticas, llamadas de la API o pruebas locales), debemos registrar la petición en `user_requests` utilizando un correo y ID por defecto.

### Cambios propuestos en `pipeline/orchestrator.py`:

```diff
def _analyze_store_embed(
    all_comments: list[dict],
    stats: dict,
    errors: list[str],
    prompt: str,
    social_medias: list[str],
    user_id: str,
    session_id: str,
    model: str,
    gemma_processed: bool,
    user_email: str | None = None,
    metrics: dict | None = None,
) -> dict:
    if not all_comments:
-       if user_email:
-           try:
-               from db.ops import insert_user_request
-               ...
-               insert_user_request(
-                   user_id=user_id,
-                   user_email=user_email,
-                   ...
-               )
-           except Exception as e:
-               logger.error(f"Error al insertar peticion vacia: {e}")
+       # Registrar petición vacía siempre (usando email/id default si es None)
+       try:
+           from db.ops import insert_user_request
+           from utils.cost_tracker import get_current_tracker
+           tracker = get_current_tracker()
+           cost = tracker.get_cost_usd() if tracker else 0.0
+           email_to_use = user_email if user_email else "anonimo@chismesitogpt.com"
+           insert_user_request(
+               user_id=user_id,
+               user_email=email_to_use,
+               session_id=session_id,
+               query=prompt,
+               social_medias=social_medias,
+               comments_count=0,
+               model=model,
+               cost=cost
+           )
+       except Exception as e:
+           logger.error(f"Error al insertar peticion vacia: {e}")

        return {
            "dataframe": pd.DataFrame(),
            "stats": stats,
            "session_id": session_id,
            "errors": errors,
            "gemma_processed": gemma_processed,
        }

    # Consolidar
    df = pd.DataFrame(all_comments)
    logger.info(f"Total consolidado: {len(df)} comentarios")

    # Analizar
    df = analyze_comments(df, model=model, metrics=metrics)

    # Insertar petición del usuario (antes de insertar comentarios para cumplir con FK)
-   if user_email:
-       try:
-           from db.ops import insert_user_request
-           ...
-           insert_user_request(
-               user_id=user_id,
-               user_email=user_email,
-               ...
-           )
-       except Exception as e:
-           logger.error(f"Error al registrar peticion de busqueda: {e}")
+   try:
+       from db.ops import insert_user_request
+       from utils.cost_tracker import get_current_tracker
+       tracker = get_current_tracker()
+       cost = tracker.get_cost_usd() if tracker else 0.0
+       email_to_use = user_email if user_email else "anonimo@chismesitogpt.com"
+       insert_user_request(
+           user_id=user_id,
+           user_email=email_to_use,
+           session_id=session_id,
+           query=prompt,
+           social_medias=social_medias,
+           comments_count=len(df),
+           model=model,
+           cost=cost
+       )
+   except Exception as e:
+       logger.error(f"Error al registrar peticion de busqueda: {e}")

    # Guardar en Supabase...
```

---

## 3. Verificación de Modelos de Embeddings (`llm_manager.py`)

Mantener el orden actual de `EMBEDDING_CANDIDATES` en [llm_manager.py](file:///E:/Users/1167486/Local/scripts/Social_media_comments/chismesito_gpt_nueva_version/llm_manager.py#L333) es correcto siempre y cuando se apliquen los cambios de dimensiones del apartado 1:
```python
EMBEDDING_CANDIDATES = [
    "models/gemini-embedding-2",    # 3072d
    "models/gemini-embedding-001",  # 768d
]
```
De esta forma, la app funcionará óptimamente con la calidad superior de `gemini-embedding-2`.
