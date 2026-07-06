# -*- coding: utf-8 -*-
"""
test_quick.py — Prueba rapida de modulos sin lanzar UI completa.

Ejecutar desde chismesito_gpt_nueva_version/:
    python test_quick.py

Prueba:
1. Config: carga de variables
2. DB: conexion Supabase
3. LLM: Gemini (si GEMINI_API_KEY configurada)
4. NLP: PySentimiento (sentiment + emotion)
5. Tools: YouTube, Reddit (si API keys configuradas)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("🧪 ChismesitoGPT v2 — Quick Test")
print("=" * 60)

# 1. Config
print("\n[1/5] Config...")
from config import (
    GEMINI_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY,
    SUPABASE_URL, SUPABASE_SERVICE_KEY,
    YOUTUBE_API_KEY, REDDIT_CLIENT_ID,
    APIFY_API_KEY, SERPAPI_API_KEY,
)
checks = {
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_KEY": SUPABASE_SERVICE_KEY,
    "YOUTUBE_API_KEY": YOUTUBE_API_KEY,
    "REDDIT_CLIENT_ID": REDDIT_CLIENT_ID,
    "APIFY_API_KEY": APIFY_API_KEY,
    "SERPAPI_API_KEY": SERPAPI_API_KEY,
}
for name, val in checks.items():
    status = "✅" if val else "❌ (no configurada)"
    print(f"  {name}: {status}")

if not GEMINI_API_KEY and not DEEPSEEK_API_KEY:
    print("\n⚠️  ADVERTENCIA: Sin API key de LLM (Gemini ni DeepSeek).")
    print("   La app no podra generar categorias ni responder en el chat.")

# 2. Supabase
print("\n[2/5] Supabase...")
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        from db.supabase_client import get_supabase_client
        client = get_supabase_client()
        print(f"  ✅ Conectado a Supabase: {SUPABASE_URL}")
    except Exception as e:
        print(f"  ❌ Error Supabase: {e}")
else:
    print("  ⏭️  Saltado (sin credenciales)")

# 3. LLM
print("\n[3/5] LLM (Gemini)...")
if GEMINI_API_KEY:
    try:
        from llm_manager import get_llm_response
        resp = get_llm_response("Responde solo con: OK", model="gemini-2.5-flash-lite")
        if "OK" in resp or "ok" in resp.lower():
            print(f"  ✅ Gemini responde: {resp.strip()}")
        else:
            print(f"  ⚠️ Respuesta inesperada: {resp.strip()[:80]}")
    except Exception as e:
        print(f"  ❌ Error Gemini: {e}")
else:
    print("  ⏭️  Saltado (sin GEMINI_API_KEY)")

# 4. NLP
print("\n[4/5] NLP (PySentimiento)...")
try:
    from tools.sentiment_tool import analyze_sentiment_text
    from tools.emotion_tool import analyze_emotion_text
    s = analyze_sentiment_text("Me encanta este producto, es excelente")
    e = analyze_emotion_text("Me encanta este producto, es excelente")
    print(f"  ✅ Sentimiento: {s} | Emocion: {e}")
except Exception as e:
    print(f"  ❌ Error NLP: {e}")

# 5. Tools (opcional)
print("\n[5/5] Tools...")
if YOUTUBE_API_KEY:
    try:
        from tools.youtube_tool import get_youtube_comments
        print("  ✅ YouTube tool importada")
    except Exception as e:
        print(f"  ❌ YouTube: {e}")
else:
    print("  ⏭️  YouTube (sin key)")

if REDDIT_CLIENT_ID:
    try:
        from tools.reddit_tool import get_reddit_comments
        print("  ✅ Reddit tool importada")
    except Exception as e:
        print(f"  ❌ Reddit: {e}")
else:
    print("  ⏭️  Reddit (sin keys)")

if APIFY_API_KEY:
    try:
        from tools.apify_tool import apify_scraper
        print("  ✅ APIFY tool importada")
    except Exception as e:
        print(f"  ❌ APIFY: {e}")
else:
    print("  ⏭️  APIFY (sin key)")

try:
    from tools.playstore_tool import get_playstore_reviews
    print("  ✅ PlayStore tool importada")
except Exception as e:
    print(f"  ❌ PlayStore: {e}")

print("\n" + "=" * 60)
print("✅ Tests completados. Ejecuta: python app.py")
print("=" * 60)
