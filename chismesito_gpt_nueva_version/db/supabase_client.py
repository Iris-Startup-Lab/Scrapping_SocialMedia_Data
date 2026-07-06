# -*- coding: utf-8 -*-
"""db/supabase_client.py — Cliente Supabase singleton (schema: chismesito_gpt)."""

from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

SCHEMA = "chismesito_gpt"
_client: Client | None = None


def get_supabase_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise RuntimeError("SUPABASE_URL y SUPABASE_SERVICE_KEY en .env")
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _client
