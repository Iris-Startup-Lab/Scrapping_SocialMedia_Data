# -*- coding: utf-8 -*-
"""
auth/supabase_auth.py — Autenticación con doble barrera para ChismesitoGPT.

Barrera 1: Whitelist de emails en ALLOWED_EMAILS (.env)
           → Sin llamada a Supabase, respuesta inmediata.
           → Mismo mensaje de error que credenciales incorrectas
             (no revela que existe whitelist).

Barrera 2: Rate limiting en memoria (5 intentos / 5 minutos por email)
           → Protege contra brute force sin necesidad de DB.

Barrera 3: Supabase Auth sign_in_with_password()
           → Valida contraseña hasheada. Retorna JWT real.
           → JWT contiene user_id (UUID de auth.users).

El user_id del JWT se usa como user_id en unified_comments y chat_history,
lo que activa correctamente las RLS policies (auth.uid() = user_id).

NOTA: No modificamos Auth Settings globales del proyecto Supabase compartido.
      La whitelist es el control de acceso propio de ChismesitoGPT.
"""

import logging
import time
import os

from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_ANON_KEY

logger = logging.getLogger(__name__)

# ─── Whitelist de emails autorizados ────────────────────────────────────────
# Cargados desde .env: ALLOWED_EMAILS=a@mail.com,b@mail.com
_raw_emails = os.getenv("ALLOWED_EMAILS", "")
ALLOWED_EMAILS: set[str] = {
    e.strip().lower() for e in _raw_emails.split(",") if e.strip()
}

if ALLOWED_EMAILS:
    logger.info(f"Auth: whitelist cargada con {len(ALLOWED_EMAILS)} email(s).")
else:
    logger.warning(
        "Auth: ALLOWED_EMAILS no configurado en .env. "
        "Cualquier usuario de Supabase Auth podrá acceder."
    )

# ─── Rate limiting en memoria ────────────────────────────────────────────────
# Formato: {"email": {"attempts": int, "last_attempt": float}}
# Se resetea cuando el servidor se reinicia (aceptable para esta escala).
_rate_registry: dict[str, dict] = {}

MAX_ATTEMPTS = 5         # intentos fallidos antes de bloquear
LOCKOUT_SECONDS = 300    # 5 minutos de bloqueo
LOGIN_DELAY = 0.5        # delay mínimo anti-timing attack (segundos)

# ─── Cliente Supabase (anon_key — solo para sign_in) ────────────────────────
# El service_key lo usa db/supabase_client.py para operaciones de backend.
_anon_client: Client | None = None


def _get_anon_client() -> Client:
    global _anon_client
    if _anon_client is None:
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            raise RuntimeError(
                "SUPABASE_URL y SUPABASE_ANON_KEY deben estar en .env"
            )
        _anon_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _anon_client


# ─── Helpers de rate limiting ────────────────────────────────────────────────

def _check_rate_limit(email: str) -> tuple[bool, str]:
    """
    Verifica si el email está bloqueado por demasiados intentos fallidos.

    Returns:
        (permitido: bool, mensaje_error: str)
    """
    now = time.time()
    rec = _rate_registry.get(email, {"attempts": 0, "last_attempt": 0.0})

    # Si pasó el tiempo de bloqueo, resetear el contador
    if now - rec["last_attempt"] > LOCKOUT_SECONDS:
        _rate_registry[email] = {"attempts": 0, "last_attempt": now}
        return True, ""

    if rec["attempts"] >= MAX_ATTEMPTS:
        wait = int(LOCKOUT_SECONDS - (now - rec["last_attempt"]))
        return False, f"⛔ Demasiados intentos fallidos. Espera {wait} segundos e inténtalo de nuevo."

    return True, ""


def _record_attempt(email: str, success: bool) -> None:
    """Registra un intento de login (exitoso o fallido) en el rate registry."""
    now = time.time()
    if success:
        # Login exitoso → resetear contador
        _rate_registry[email] = {"attempts": 0, "last_attempt": now}
    else:
        rec = _rate_registry.get(email, {"attempts": 0, "last_attempt": now})
        rec["attempts"] = rec.get("attempts", 0) + 1
        rec["last_attempt"] = now
        _rate_registry[email] = rec


# ─── Función principal de login ──────────────────────────────────────────────

def sign_in(email: str, password: str) -> dict:
    """
    Intenta hacer login con doble barrera: whitelist + Supabase Auth.

    Args:
        email:    Email del usuario (se normaliza a lowercase).
        password: Contraseña en texto plano (Supabase la verifica con bcrypt).

    Returns:
        {
            "success":      bool,
            "user_id":      str | None,   # UUID de auth.users
            "email":        str | None,   # email normalizado
            "access_token": str | None,   # JWT para RLS
            "error":        str | None    # mensaje de error (nunca expone internals)
        }
    """
    # ── Validación básica ────────────────────────────────────────────────────
    if not email or not email.strip():
        return _error("❌ Ingresa tu email.")
    if not password:
        return _error("❌ Ingresa tu contraseña.")

    email = email.strip().lower()

    # ── Barrera 1: Whitelist ─────────────────────────────────────────────────
    if ALLOWED_EMAILS and email not in ALLOWED_EMAILS:
        logger.warning(f"Auth: acceso denegado (email no en whitelist): {email}")
        # Usamos el mismo mensaje que credenciales incorrectas
        # para no revelar que existe una whitelist.
        time.sleep(LOGIN_DELAY)
        return _error("❌ Credenciales incorrectas.")

    # ── Barrera 2: Rate limiting ─────────────────────────────────────────────
    allowed, lock_msg = _check_rate_limit(email)
    if not allowed:
        return _error(lock_msg)

    # Delay mínimo anti-timing attack y anti-brute force
    time.sleep(LOGIN_DELAY)

    # ── Barrera 3: Supabase Auth ─────────────────────────────────────────────
    try:
        resp = _get_anon_client().auth.sign_in_with_password(
            {"email": email, "password": password}
        )

        if not resp.user or not resp.session:
            _record_attempt(email, False)
            return _error("❌ Credenciales incorrectas.")

        _record_attempt(email, True)
        logger.info(f"Auth: login exitoso → {email} (user_id: {resp.user.id})")

        return {
            "success":      True,
            "user_id":      str(resp.user.id),
            "email":        resp.user.email,
            "access_token": resp.session.access_token,
            "error":        None,
        }

    except Exception as e:
        _record_attempt(email, False)
        # Logeamos el error real pero NO lo exponemos al usuario
        logger.warning(f"Auth: Supabase sign_in error para {email}: {e}")
        return _error("❌ Credenciales incorrectas.")


def sign_out() -> bool:
    """
    Cierra la sesión en Supabase (best-effort).
    El estado del usuario en app.py se resetea independientemente del resultado.
    """
    try:
        _get_anon_client().auth.sign_out()
        logger.info("Auth: sign_out exitoso.")
        return True
    except Exception as e:
        logger.warning(f"Auth: sign_out error (ignorado): {e}")
        return True  # El logout local siempre es exitoso


def get_login_attempts_info(email: str) -> dict:
    """
    Retorna información del estado de rate limiting para un email.
    Útil para debugging y para mostrar feedback en la UI.
    """
    email = email.strip().lower()
    rec = _rate_registry.get(email, {"attempts": 0, "last_attempt": 0.0})
    remaining = max(0, MAX_ATTEMPTS - rec.get("attempts", 0))
    return {
        "attempts":  rec.get("attempts", 0),
        "remaining": remaining,
        "locked":    rec.get("attempts", 0) >= MAX_ATTEMPTS,
    }


# ─── Helper interno ──────────────────────────────────────────────────────────

def _error(msg: str) -> dict:
    """Retorna un dict de error estandarizado."""
    return {
        "success":      False,
        "user_id":      None,
        "email":        None,
        "access_token": None,
        "error":        msg,
    }
