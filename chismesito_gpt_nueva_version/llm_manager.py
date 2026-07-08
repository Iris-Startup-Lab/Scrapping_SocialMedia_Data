# -*- coding: utf-8 -*-
"""
llm_manager.py — Multi-provider LLM con APIs oficiales:
  - Gemini:      google-genai SDK
  - DeepSeek:    OpenAI-compatible (https://api.deepseek.com)
  - Claude:      Anthropic Messages API (https://api.anthropic.com)

Features:
  - Llamada unificada get_llm_response(model, prompt)
  - Listado dinámico de modelos disponibles (list_available_models)
  - Filtro de modelos prohibidos (caros / ineficientes)
  - Pricing por modelo para mostrar en UI
"""

import logging
import json
import requests
from typing import Literal, Any
from config import GEMINI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY

logger = logging.getLogger(__name__)

# ─── Modelos prohibidos (caros o ineficientes) ──────────────────────
PROHIBITED_MODELS: set[str] = {
    # Claude — modelos legacy/deprecados o ultra-caros
    "claude-opus-4-1-20250805",   # Deprecado/caro, $15/$75 por MTok
    "claude-opus-4-5-20251101",   # Legacy
    "claude-fable-5",             # $10/$50 por MTok — overkill
    "claude-mythos-5",            # Solo por invitación
    # DeepSeek — modelos deprecados
    "deepseek-chat",              # Deprecado 2026/07/24
    "deepseek-reasoner",          # Deprecado 2026/07/24
    # Gemini — modelos legacy / deprecados
    "gemini-1.0-pro",             # Muy antiguo
    "gemini-1.5-pro",             # Legacy
    "gemini-1.5-flash",           # Legacy
    "gemini-2.0-flash-lite",      # Sin sufijo -001, da 404 NOT_FOUND
}

# ─── Catálogo estático de modelos (fallback si API no disponible) ───
STATIC_MODELS = [
    # DeepSeek — precios verificados julio 2026
    {"id": "deepseek-v4-flash",    "provider": "deepseek", "label": "DeepSeek V4 Flash (económico)",
     "pricing": "$0.14 / $0.28 por MTok", "context": "1M tokens", "output": "384K tokens"},
    {"id": "deepseek-v4-pro",      "provider": "deepseek", "label": "DeepSeek V4 Pro (potente)",
     "pricing": "$0.435 / $0.87 por MTok", "context": "1M tokens", "output": "384K tokens"},
    # Gemini — precios verificados julio 2026 (tier de pago)
    {"id": "gemini-2.5-flash-lite",     "provider": "google", "label": "Gemini 2.5 Flash Lite",
     "pricing": "$0.10 / $0.40 por MTok",  "context": "1M tokens",  "output": "16384 tokens"},
    {"id": "gemini-2.5-flash",          "provider": "google", "label": "Gemini 2.5 Flash",
     "pricing": "$0.30 / $2.50 por MTok",  "context": "1M tokens",  "output": "65536 tokens"},
    {"id": "gemini-2.5-pro",            "provider": "google", "label": "Gemini 2.5 Pro (max calidad)",
     "pricing": "$1.25 / $10 por MTok",    "context": "2M tokens",  "output": "65536 tokens"},
    {"id": "gemini-3.1-flash-lite",     "provider": "google", "label": "Gemini 3.1 Flash Lite",
     "pricing": "$0.25 / $1.50 por MTok",  "context": "1M tokens",  "output": "16384 tokens"},
    {"id": "gemini-3.5-flash",          "provider": "google", "label": "Gemini 3.5 Flash (último)",
     "pricing": "$1.50 / $9.00 por MTok",  "context": "1M tokens",  "output": "65536 tokens"},
    {"id": "gemma-4-31b-it",            "provider": "google", "label": "Gemma 4 31B IT (razonamiento)",
     "pricing": "$0.07 / $0.07 por MTok",  "context": "8K tokens",   "output": "8192 tokens"},
    # Claude (Anthropic) — precios verificados julio 2026
    {"id": "claude-haiku-4-5-20251001",  "provider": "anthropic", "label": "Claude Haiku 4.5 (rápido)",
     "pricing": "$0.80 / $4 por MTok",   "context": "200K tokens", "output": "16K tokens"},
    {"id": "claude-sonnet-4-5-20250929", "provider": "anthropic", "label": "Claude Sonnet 4.5",
     "pricing": "$3 / $15 por MTok",     "context": "200K tokens", "output": "64K tokens"},
    {"id": "claude-sonnet-4-6",          "provider": "anthropic", "label": "Claude Sonnet 4.6",
     "pricing": "$3 / $15 por MTok",     "context": "1M tokens",   "output": "128K tokens"},
    {"id": "claude-opus-4-8",            "provider": "anthropic", "label": "Claude Opus 4.8 (más capaz)",
     "pricing": "$5 / $25 por MTok",     "context": "1M tokens",   "output": "128K tokens"},
    # OpenAI — chat estándar (no-razonamiento), compatibles con temperature + max_completion_tokens
    {"id": "gpt-4o-mini",         "provider": "openai", "label": "GPT-4o mini (económico)",
     "pricing": "$0.15 / $0.60 por MTok", "context": "128K tokens", "output": "16K tokens"},
    {"id": "gpt-4o",              "provider": "openai", "label": "GPT-4o",
     "pricing": "$2.50 / $10 por MTok",   "context": "128K tokens", "output": "16K tokens"},
    {"id": "gpt-5-chat-latest",   "provider": "openai", "label": "GPT-5 Chat (último)",
     "pricing": "$1.25 / $10 por MTok",   "context": "400K tokens", "output": "128K tokens"},
]

# ─── Funciones de llamada a APIs oficiales ─────────────────────────

def _call_gemini(prompt: str, model: str, system_prompt: str | None = None) -> tuple[str, dict]:
    """Gemini nativo via google-genai SDK. Retorna (texto, usage)."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY no configurada")
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    if system_prompt:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"system_instruction": system_prompt},
        )
    else:
        response = client.models.generate_content(model=model, contents=prompt)
    usage = {"in": 0, "out": 0}
    um = getattr(response, "usage_metadata", None)
    if um is not None:
        usage["in"] = getattr(um, "prompt_token_count", 0) or 0
        usage["out"] = getattr(um, "candidates_token_count", 0) or 0
    return response.text, usage


def _call_deepseek(prompt: str, model: str, system_prompt: str | None = None) -> tuple[str, dict]:
    """DeepSeek via API OpenAI-compatible (https://api.deepseek.com). Retorna (texto, usage)."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY no configurada")
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=2048,
        temperature=0.3,
    )
    usage = {"in": 0, "out": 0}
    u = getattr(response, "usage", None)
    if u is not None:
        usage["in"] = getattr(u, "prompt_tokens", 0) or 0
        usage["out"] = getattr(u, "completion_tokens", 0) or 0
    return response.choices[0].message.content, usage


def _call_openai(prompt: str, model: str, system_prompt: str | None = None) -> tuple[str, dict]:
    """OpenAI via API oficial (https://api.openai.com). Retorna (texto, usage).
    Usa max_completion_tokens (compatible con gpt-4o y gpt-5-chat)."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no configurada")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=2048,
        temperature=0.3,
    )
    usage = {"in": 0, "out": 0}
    u = getattr(response, "usage", None)
    if u is not None:
        usage["in"] = getattr(u, "prompt_tokens", 0) or 0
        usage["out"] = getattr(u, "completion_tokens", 0) or 0
    return response.choices[0].message.content, usage


def _call_claude(prompt: str, model: str, system_prompt: str | None = None) -> tuple[str, dict]:
    """Claude via Anthropic Messages API (https://api.anthropic.com). Retorna (texto, usage)."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY no configurada")

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 2048,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        payload["system"] = system_prompt

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    u = data.get("usage", {}) or {}
    usage = {"in": u.get("input_tokens", 0) or 0, "out": u.get("output_tokens", 0) or 0}
    return data["content"][0]["text"], usage


# ─── Mapeo provider → función ───────────────────────────────────────
PROVIDER_CALL = {
    "google":    _call_gemini,
    "deepseek":  _call_deepseek,
    "anthropic": _call_claude,
    "openai":    _call_openai,
}

PROVIDER_OF = {m["id"]: m["provider"] for m in STATIC_MODELS}


def get_llm_response(
    prompt: str,
    model: str = "gemini-2.5-flash",
    system_prompt: str | None = None,
) -> str:
    """
    Envía un prompt al modelo y retorna la respuesta.
    Usa automáticamente la API oficial según el provider del modelo.
    """
    if model in PROHIBITED_MODELS:
        return f"Error: El modelo '{model}' esta en la lista de modelos prohibidos (caro o ineficiente)."

    provider = PROVIDER_OF.get(model)
    if not provider:
        return f"Error: Modelo '{model}' no reconocido. Usa list_available_models() para ver opciones."

    call_fn = PROVIDER_CALL.get(provider)
    if not call_fn:
        return f"Error: Provider '{provider}' no soportado."

    try:
        text, usage = call_fn(prompt, model, system_prompt)
        # Registrar costo en el tracker activo (si hay uno) con tokens reales.
        try:
            from utils.cost_tracker import record_llm
            record_llm(model, usage.get("in", 0), usage.get("out", 0))
        except Exception as e:
            logger.debug(f"No se pudo registrar costo LLM: {e}")
        return text
    except Exception as e:
        logger.error(f"LLM call failed ({model}/{provider}): {e}")
        return f"Error al llamar a {model}: {str(e)[:200]}"


# ─── Listado de modelos (dinámico desde APIs) ──────────────────────

def list_available_models() -> list[dict]:
    """
    Retorna lista de modelos disponibles, combinando:
      1. API de Anthropic (GET /v1/models) si ANTHROPIC_API_KEY existe
      2. API de DeepSeek  (desde catálogo estático actualizado)
      3. Gemini           (desde catálogo estático)
    
    Filtra modelos prohibidos.
    Retorna lista de dicts: {id, provider, label, pricing, context, output}
    """
    available = []

    # 1. Claude models desde API oficial
    if ANTHROPIC_API_KEY:
        try:
            resp = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                },
                timeout=15,
            )
            if resp.ok:
                api_data = resp.json().get("data", [])
                claude_ids = {m["id"] for m in api_data}
                logger.info(f"Anthropic API IDs reales: {sorted(claude_ids)}")
                matched = [m for m in STATIC_MODELS if m["provider"] == "anthropic" and m["id"] in claude_ids]
                if matched:
                    available.extend(matched)
                    logger.info(f"Anthropic API: {len(matched)} modelos del catalogo disponibles")
                else:
                    # Los IDs del catálogo no coinciden con la API → usar catálogo completo
                    logger.warning(
                        f"Anthropic: ningún ID del catálogo coincidió con la API "
                        f"(API IDs: {sorted(claude_ids)[:5]}...). Usando catálogo estático."
                    )
                    available.extend([m for m in STATIC_MODELS if m["provider"] == "anthropic"])
            else:
                logger.warning(f"Anthropic models API: {resp.status_code}")
                available.extend([m for m in STATIC_MODELS if m["provider"] == "anthropic"])
        except Exception as e:
            logger.warning(f"Anthropic models API error: {e}")
            available.extend([m for m in STATIC_MODELS if m["provider"] == "anthropic"])

    # 2. DeepSeek — catálogo estático (la API no expone endpoint /models público)
    #    Nota: deepseek-v4-flash y deepseek-v4-pro se verifican contra pricing oficial
    if DEEPSEEK_API_KEY:
        available.extend([m for m in STATIC_MODELS if m["provider"] == "deepseek"])

    # 3. Gemini — catálogo estático (disponible si GEMINI_API_KEY existe)
    if GEMINI_API_KEY:
        available.extend([m for m in STATIC_MODELS if m["provider"] == "google"])

    # 4. OpenAI — catálogo estático (disponible si OPENAI_API_KEY existe)
    if OPENAI_API_KEY:
        available.extend([m for m in STATIC_MODELS if m["provider"] == "openai"])

    # 5. Filtrar prohibidos
    result = [m for m in available if m["id"] not in PROHIBITED_MODELS]

    # Si no hay nada, devolver catálogo completo (sin keys configuradas)
    if not result:
        result = [m for m in STATIC_MODELS if m["id"] not in PROHIBITED_MODELS]

    # Ordenar por provider: deepseek → google → anthropic → openai
    order = {"deepseek": 0, "google": 1, "anthropic": 2, "openai": 3}
    result.sort(key=lambda m: order.get(m["provider"], 99))

    return result


# ─── Helper: generar categorías con modelo seleccionado ─────────────
def generate_categories_with_model(
    comments_text: str,
    model: str = "gemini-2.5-flash",
) -> list[str]:
    """Genera categorías usando el modelo seleccionado."""
    import ast
    max_len = 7000
    text = comments_text[:max_len] if len(comments_text) > max_len else comments_text

    system = "Eres un analista experto. Responde EXCLUSIVAMENTE con una lista de Python."
    prompt = f"""Analiza los siguientes comentarios de redes sociales:

{text}

Genera entre 5 y 10 categorias tematicas (maximo 4 palabras cada una) que
resuman los temas principales. Cada categoria debe ser una etiqueta breve.

IMPORTANTE:
- TODAS las categorias DEBEN estar en ESPAÑOL, sin importar el idioma de los comentarios.
- Responde SOLO con una lista de Python. Ejemplo:
['Calidad del producto', 'Precio', 'Servicio al cliente', 'Entrega', 'Empaque']"""

    raw = (get_llm_response(prompt=prompt, model=model, system_prompt=system) or "").strip()

    # get_llm_response devuelve un STRING que empieza con "Error" si la llamada falló
    # (timeout, rate-limit, modelo no disponible…). No lo parsees como categorías:
    # regresa vacío para que el caller aplique su fallback genérico, y déjalo LOGUEADO.
    if raw.startswith("Error"):
        logger.warning(f"generate_categories: el LLM ({model}) no respondió con categorías. Detalle: {raw[:180]}")
        return []

    # Limpiar bloques de código markdown que el LLM puede devolver
    import re
    raw = re.sub(r"```(?:python)?\s*", "", raw)
    raw = re.sub(r"```", "", raw)
    raw = raw.strip()

    # 1) Intento directo: la salida ES una lista de Python
    try:
        categories = ast.literal_eval(raw)
        if isinstance(categories, list):
            cats = [str(c).strip() for c in categories if str(c).strip()]
            if cats:
                return cats[:10]
    except (SyntaxError, ValueError):
        pass

    # 2) Intento tolerante: extraer el primer [...] aunque venga rodeado de texto
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            categories = ast.literal_eval(m.group(0))
            if isinstance(categories, list):
                cats = [str(c).strip().strip("'\"") for c in categories if str(c).strip()]
                if cats:
                    return cats[:10]
        except (SyntaxError, ValueError):
            pass

    # No se pudo parsear: LOGUEAR la salida cruda (para diagnóstico) y dejar que
    # el caller use su fallback genérico en vez de inventar categorías basura.
    logger.warning(f"generate_categories: no pude parsear categorías del LLM ({model}). "
                   f"Salida cruda: {raw[:200]!r}")
    return []


# ─── Información de configuración para UI ───────────────────────────
def get_provider_status() -> dict[str, dict]:
    """Retorna estado de cada provider (conectado o no)."""
    return {
        "gemini": {
            "name": "Google Gemini",
            "configured": bool(GEMINI_API_KEY),
            "api_doc": "https://ai.google.dev/gemini-api/docs",
        },
        "deepseek": {
            "name": "DeepSeek",
            "configured": bool(DEEPSEEK_API_KEY),
            "api_doc": "https://api-docs.deepseek.com/",
        },
        "anthropic": {
            "name": "Anthropic Claude",
            "configured": bool(ANTHROPIC_API_KEY),
            "api_doc": "https://docs.anthropic.com/en/api/",
        },
        "openai": {
            "name": "OpenAI",
            "configured": bool(OPENAI_API_KEY),
            "api_doc": "https://platform.openai.com/docs/api-reference",
        },
    }


# ─── Embedding model discovery ──────────────────────────────────────

EMBEDDING_CANDIDATES = [
    # Modelos verificados con la API key actual (2026-06-29)
    "models/gemini-embedding-2",    # Mas reciente, 3072d
    "models/gemini-embedding-001",  # Estable, 768d (compatible con pgvector)
]

def discover_embedding_model() -> str | None:
    """
    Prueba modelos de embedding hasta encontrar uno funcional en Gemini API.
    El SDK google-genai 2.8.0 usa v1beta por defecto; los embeddings requieren v1.
    Se prueba primero con api_version='v1', luego con v1beta como fallback.
    """
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        from google.genai import types as genai_types

        # Cliente con v1 (requerido por los modelos de embedding en SDK 2.8+)
        try:
            client_v1 = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options={"api_version": "v1"},
            )
        except Exception:
            client_v1 = None

        # Cliente default (v1beta)
        client_default = genai.Client(api_key=GEMINI_API_KEY)

        for model in EMBEDDING_CANDIDATES:
            for client in ([client_v1] if client_v1 else []) + [client_default]:
                if client is None:
                    continue
                try:
                    result = client.models.embed_content(
                        model=model,
                        contents="test",   # string, no lista
                    )
                    if hasattr(result, "embeddings") and result.embeddings:
                        logger.info(f"Embedding model OK: {model}")
                        return model
                except Exception:
                    try:
                        # Fallback: contents como lista
                        result = client.models.embed_content(
                            model=model,
                            contents=["test"],
                        )
                        if hasattr(result, "embeddings") and result.embeddings:
                            logger.info(f"Embedding model OK (lista): {model}")
                            return model
                    except Exception:
                        continue
        logger.warning("Ningun embedding model funciono")
        return None
    except Exception as e:
        logger.error(f"Error discovering embedding model: {e}")
        return None
