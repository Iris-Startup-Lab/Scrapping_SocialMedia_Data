# -*- coding: utf-8 -*-
"""config.py — Carga centralizada de variables de entorno."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# LLMs — APIs oficiales
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Search
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# APIs Oficiales
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
MAPS_API_KEY = os.getenv("MAPS_API_KEY")
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "chismesito_gpt_v2/1.0")

# APIFY
APIFY_API_KEY = os.getenv("APIFY_API_KEY")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# App
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Detección de entorno HuggingFace Spaces (HF define SPACE_ID en sus contenedores).
# Sirve para activar la ruta GPU (ZeroGPU) y omitir comportamientos locales.
IS_SPACES = "SPACE_ID" in os.environ

# Auth — whitelist de emails autorizados (CSV en .env)
# Ejemplo: ALLOWED_EMAILS=admin@empresa.com,colega@empresa.com
ALLOWED_EMAILS_RAW = os.getenv("ALLOWED_EMAILS", "")

# Embedding
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 3072  # gemini-embedding-2 usa 3072d
