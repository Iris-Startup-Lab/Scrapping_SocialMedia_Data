# -*- coding: utf-8 -*-
"""app.py — ChismesitoGPT v2 — Gradio Entry Point."""

import logging, uuid, tempfile, os, signal, subprocess, sys, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette")
from pathlib import Path
import pandas as pd
import gradio as gr
from pipeline.orchestrator import run_pipeline, discover, run_pipeline_from_selection
from tools.google_maps_tool import geocode_location
from pipeline.rag import rag_chat
from ui.dashboard import (plot_sentiment, plot_emotions, plot_categories,
                          plot_platform_counts, plot_map, plot_places_preview)
from ui.styles import (CUSTOM_CSS, get_model_badge_html, get_social_selector_html,
                       get_selection_cards_html, get_kpi_cards_html,
                       get_analysis_summary_html, SOCIAL_MEDIA_DEFS, SOCIAL_ICONS)
from llm_manager import (list_available_models, get_provider_status, PROHIBITED_MODELS,
                         PROVIDER_OF, discover_embedding_model)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for name in ("httpx","httpcore","gradio","urllib3","asyncio"):
    logging.getLogger(name).setLevel(logging.WARNING)

AVAILABLE_MODELS = list_available_models()
# Usar DeepSeek V4 Flash como default (más económico), con fallbacks
_preferred = ["deepseek-v4-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash"]
DEFAULT_MODEL = next(
    (m["id"] for p in _preferred for m in AVAILABLE_MODELS if m["id"] == p),
    AVAILABLE_MODELS[0]["id"] if AVAILABLE_MODELS else "deepseek-v4-flash"
)
MODEL_CHOICES    = [(f"{m['label']} | {m['pricing']}", m["id"]) for m in AVAILABLE_MODELS]
PROVIDER_STATUS  = get_provider_status()
EMB_MODEL        = discover_embedding_model()
logger.info(f"Embedding: {'✅ '+EMB_MODEL if EMB_MODEL else '❌ No detectado'}")

class SessionState:
    """Estado por sesión de usuario. Se instancia una vez y se pasa como gr.State."""
    def __init__(self):
        self.df           = pd.DataFrame()
        self.stats        = {}
        self.session_id   = str(uuid.uuid4())
        self.errors       = []
        self.chat_history = []
        self.discovery    = {}   # Fase 1: {platform: [Item,...]} para el flujo manual
        # ── Auth ──────────────────────────────────────
        self.is_logged_in = False
        self.user_id      = None   # UUID real de auth.users (None hasta login)
        self.user_email   = None   # Email del usuario logueado
        self.access_token = None   # JWT de Supabase
        # ── App ──────────────────────────────────────
        self.model        = DEFAULT_MODEL
        # ── Cost tracker ──────────────────────────────
        from utils.cost_tracker import CostTracker
        self.cost_tracker = CostTracker()

def _build_badge(model_id: str) -> str:
    provider = PROVIDER_OF.get(model_id, "google")
    info = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
    return get_model_badge_html(model_id, info["label"] if info else model_id, provider)

def _build_provider_html() -> str:
    pills = []
    for k, v in PROVIDER_STATUS.items():
        ok, dot = v["configured"], "●" if v["configured"] else "○"
        pills.append(f'<span class="provider-pill {"ok" if ok else "off"}">{dot} {v["name"]}</span>')
    return '<div class="provider-pills">' + "".join(pills) + "</div>"

def _build_user_header(email: str) -> str:
    """HTML del header cuando el usuario está logueado."""
    initials = email[0].upper() if email else "U"
    return f"""
<div class="user-header" style="display:flex;align-items:center;gap:10px;padding:8px 14px;
     background:rgba(13,148,136,0.1);border:1px solid rgba(13,148,136,0.25);
     border-radius:12px;margin-bottom:4px;">
  <div style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#0d9488,#0891b2);
       display:flex;align-items:center;justify-content:center;font-weight:700;color:#fff;font-size:14px;">
    {initials}
  </div>
  <span style="color:#5eead4;font-size:13px;font-weight:500;">{email}</span>
  <span style="margin-left:auto;font-size:11px;opacity:0.5;">Sesión activa</span>
</div>"""

def _build_model_banner(model_id: str) -> str:
    """Banner informativo del modelo activo, se muestra junto al prompt."""
    info  = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
    label = info["label"] if info else model_id
    price = info["pricing"] if info else ""
    return (
        f'<div style="display:flex;align-items:center;gap:10px;padding:8px 14px;'
        f'background:rgba(13,148,136,0.07);border:1px solid rgba(13,148,136,0.2);'
        f'border-radius:10px;margin-top:6px;">'
        f'<span style="font-size:12px;color:rgba(255,255,255,0.35);font-family:Inter,sans-serif;">'
        f'&#x1F9E0; Modelo activo:</span>'
        f'<code style="font-size:12px;color:#5eead4;">{label}</code>'
        f'<span style="font-size:11px;color:rgba(255,255,255,0.3);">{price}</span>'
        f'<span style="margin-left:auto;font-size:11px;color:rgba(255,255,255,0.3);font-family:Inter,sans-serif;">'
        f'&#x2193; Cambia el modelo al final de la p&aacute;gina</span>'
        f'</div>'
    )

def _get_logo_html() -> str:
    """Intenta cargar el logo PNG en base64. Si falla, retorna el emoji detective."""
    logo_path = Path(__file__).parent / "icons" / "logo" / "Logo_Chismesito_GPT.png"
    if logo_path.exists():
        import base64
        try:
            b64_str = base64.b64encode(logo_path.read_bytes()).decode()
            return f'<img src="data:image/png;base64,{b64_str}" style="width: 300px; height: auto; margin: 0 auto 12px; display: block;" alt="ChismesitoGPT" />'
        except Exception:
            pass
    return '<div style="font-size: 52px; margin-bottom: 12px; line-height: 1;">🕵️</div>'

INITIAL_BADGE = _build_badge(DEFAULT_MODEL)

# ─── Handlers ───────────────────────────────────────────────────────

# ── Auth handlers ──────────────────────────────────────────────────

def handle_login(email: str, password: str, state: SessionState):
    """
    Intenta login con Supabase Auth (doble barrera: whitelist + Supabase).
    Retorna el state actualizado + actualizaciones de visibilidad de paneles.
    """
    from auth.supabase_auth import sign_in
    result = sign_in(email, password)

    if result["success"]:
        state.is_logged_in = True
        state.user_id      = result["user_id"]
        state.user_email   = result["email"]
        state.access_token = result["access_token"]
        logger.info(f"Login OK en UI: {result['email']} (user_id: {result['user_id']})")
        return (
            state,
            gr.update(visible=False),           # login_panel oculto
            gr.update(visible=True),            # app_panel visible
            "",                                 # login_status vacío
            _build_user_header(result["email"]),# user_header_html
        )
    else:
        logger.warning(f"Login fallido desde UI: {email}")
        return (
            state,
            gr.update(visible=True),    # login_panel sigue visible
            gr.update(visible=False),   # app_panel oculto
            result["error"],            # mostrar error en login_status
            "",                         # user_header_html vacío
        )


def handle_logout(state: SessionState):
    """Cierra la sesión y resetea el estado del usuario."""
    from auth.supabase_auth import sign_out
    email = state.user_email or "?"
    sign_out()
    # Crear estado limpio (no reusar el mismo objeto)
    new_state = SessionState()
    logger.info(f"Logout: {email}")
    return (
        new_state,
        gr.update(visible=True),    # login_panel visible
        gr.update(visible=False),   # app_panel oculto
        "",                         # limpiar login_email
        "",                         # limpiar login_password
        "",                         # limpiar login_status
        "",                         # limpiar user_header_html
    )


# ── Search / Chat handlers ───────────────────────────────────────────────

def _render_results(result, state, social_medias):
    """Convierte el dict de run_pipeline(_from_selection) en las salidas de la UI.

    Devuelve la 11-tupla: (sentiment, emotion, category, count, map, summary,
    display_df, csv, xlsx, state, cost). Reutilizado por el flujo automático y
    el manual para no duplicar el render de dashboards.
    """
    state.df, state.stats, state.errors = result["dataframe"], result["stats"], result["errors"]

    _show_cols = [c for c in state.df.columns if c not in ("embedding", "id", "latitude", "longitude")]
    display_df = state.df[_show_cols] if not state.df.empty else state.df

    sp = plot_sentiment(state.df) if not state.df.empty else None
    ep = plot_emotions(state.df)  if not state.df.empty else None
    cp = plot_categories(state.df) if not state.df.empty else None
    kp = plot_platform_counts(state.stats) if state.stats else None
    mp = plot_map(state.df) if not state.df.empty else None
    csv_path = xlsx_path = None
    if not state.df.empty:
        cols = [c for c in ["social_media", "username", "comment", "sentiment", "emotion", "category"] if c in state.df.columns]
        edf = state.df[cols].copy()
        csv_path  = Path(tempfile.gettempdir()) / f"chismesito_{state.session_id[:8]}.csv"
        xlsx_path = Path(tempfile.gettempdir()) / f"chismesito_{state.session_id[:8]}.xlsx"
        edf.to_csv(csv_path, index=False, encoding="utf-8-sig")
        edf.to_excel(xlsx_path, index=False, engine="openpyxl")
    total = sum(state.stats.values())
    summary = f"✅ **{total} comentarios** en {len(social_medias)} plataforma(s).\n\n"
    for p, c in state.stats.items():
        summary += f"- **{p}**: {c}\n"
    if state.errors:
        summary += f"\n⚠️ {', '.join(state.errors[:3])}"

    gemma_status = "Esta petición se procesó con Gemma" if result.get("gemma_processed") else "Petición no formateada"
    summary += f"\n\n🤖 **Procesamiento:** {gemma_status}"
    summary += f"\n🧠 `{EMB_MODEL or 'sin embedding'}`"

    return (sp, ep, cp, kp, mp, summary, display_df,
            str(csv_path) if csv_path else None,
            str(xlsx_path) if xlsx_path else None,
            state, state.cost_tracker.render(state.model))


def handle_search(prompt, social_raw, model_choice, max_comments, maps_geo_toggle, maps_location, maps_radius, state, progress=gr.Progress(track_tqdm=False)):
    # ── Guard: verificar login ──────────────────────────────────────────
    if not state.is_logged_in or not state.user_id:
        yield None, None, None, None, None, "⛔ Debes iniciar sesión para usar la app.", pd.DataFrame(), None, None, state, ""
        return

    social_medias = [v.strip() for v in (social_raw or "youtube").split(",") if v.strip() in SOCIAL_ALL]
    if not social_medias:
        social_medias = ["youtube"]

    max_comments = int(max_comments) if max_comments else 10

    if not prompt:
        yield None, None, None, None, None, "⚠️ Ingresa un prompt.", pd.DataFrame(), None, None, state, state.cost_tracker.render(state.model)
        return
    state.model = model_choice
    # Activar el tracker de costos de esta sesión para todo el pipeline.
    from utils.cost_tracker import set_current_tracker
    set_current_tracker(state.cost_tracker)
    progress(0.05, desc="Buscando...")
    result = run_pipeline(prompt=prompt, social_medias=social_medias,
                          user_id=state.user_id, session_id=state.session_id,
                          model=model_choice, max_comments=max_comments,
                          maps_geo_toggle=maps_geo_toggle,
                          maps_location=maps_location,
                          maps_radius=int(maps_radius),
                          user_email=state.user_email)
    progress(0.80, desc="Graficos...")
    outputs = _render_results(result, state, social_medias)
    progress(1.0, desc="Listo!")
    yield outputs


def handle_discover(prompt, social_raw, model_choice, max_comments,
                    maps_geo_toggle, maps_location, maps_radius, state,
                    progress=gr.Progress(track_tqdm=False)):
    """FASE 1 (modo manual): busca items candidatos y muestra las tarjetas."""
    if not state.is_logged_in or not state.user_id:
        return ("<div class='status-md'>⛔ Debes iniciar sesión.</div>",
                gr.update(visible=False), "⛔ Inicia sesión para usar la app.", state,
                state.cost_tracker.render(state.model), gr.update(visible=False))

    social_medias = [v.strip() for v in (social_raw or "youtube").split(",") if v.strip() in SOCIAL_ALL]
    if not social_medias:
        social_medias = ["youtube"]
    if not prompt:
        return ("<div class='status-md'>⚠️ Ingresa un prompt.</div>",
                gr.update(visible=False), "⚠️ Ingresa un prompt.", state,
                state.cost_tracker.render(state.model), gr.update(visible=False))

    state.model = model_choice
    from utils.cost_tracker import set_current_tracker
    set_current_tracker(state.cost_tracker)

    progress(0.2, desc="Buscando posts/videos...")
    result = discover(prompt=prompt, social_medias=social_medias,
                      maps_geo_toggle=maps_geo_toggle, maps_location=maps_location,
                      maps_radius=int(maps_radius), max_items=int(max_comments) if max_comments else 8)
    state.discovery = result.get("items", {})

    cards_html = get_selection_cards_html(state.discovery)
    found = sum(len(v) for v in state.discovery.values())
    status = f"🎯 **{found} resultados** encontrados. Elige de cuáles extraer comentarios y pulsa **Obtener comentarios**."
    notes = list(result.get("errors", []))
    unsupported = result.get("unsupported", [])
    if unsupported:
        notes.append(f"Selección manual aún no disponible para: {', '.join(unsupported)} (usa el modo automático)")
    if notes:
        status += "\n\n⚠️ " + " · ".join(notes[:4])
    # La búsqueda de Fase 1 ya consumió Apify (X, Google search de Play Store);
    # reflejamos ese costo en la calculadora aunque aún no haya comentarios.
    status += "\n\n💰 *El costo de esta búsqueda ya se refleja en la calculadora de arriba.*"

    # Vista previa de mapa para Google Maps (pines numerados = tarjetas de lugares).
    maps_places = state.discovery.get("maps", [])
    map_fig = plot_places_preview(maps_places) if maps_places else None
    map_update = gr.update(visible=bool(map_fig), value=map_fig)

    progress(1.0, desc="Listo!")
    return cards_html, gr.update(visible=True), status, state, state.cost_tracker.render(state.model), map_update


def handle_fetch_selected(prompt, selection_raw, model_choice, max_comments, state,
                          progress=gr.Progress(track_tqdm=False)):
    """FASE 2 (modo manual): extrae comentarios solo de los items seleccionados."""
    if not state.is_logged_in or not state.user_id:
        yield (None, None, None, None, None, "⛔ Debes iniciar sesión.",
               pd.DataFrame(), None, None, state, state.cost_tracker.render(state.model))
        return

    import json as _json
    try:
        selections = _json.loads(selection_raw) if selection_raw else {}
    except (ValueError, TypeError):
        selections = {}

    # Validar contra lo descubierto (solo ids realmente ofrecidos en la Fase 1).
    valid_ids = {p: {str(it.get("id")) for it in items}
                 for p, items in (state.discovery or {}).items()}
    selections = {p: [i for i in ids if i in valid_ids.get(p, set())]
                  for p, ids in selections.items()}
    selections = {p: ids for p, ids in selections.items() if ids}

    if not selections:
        yield (None, None, None, None, None,
               "⚠️ No seleccionaste ningún post/video. Marca al menos uno.",
               pd.DataFrame(), None, None, state, state.cost_tracker.render(state.model))
        return

    state.model = model_choice
    from utils.cost_tracker import set_current_tracker
    set_current_tracker(state.cost_tracker)

    progress(0.1, desc="Extrayendo comentarios de tu selección...")
    result = run_pipeline_from_selection(
        prompt=prompt, selections=selections, user_id=state.user_id,
        session_id=state.session_id, model=model_choice,
        max_comments=int(max_comments) if max_comments else 10,
        discovery=state.discovery,
        user_email=state.user_email,
    )
    progress(0.85, desc="Graficos...")
    outputs = _render_results(result, state, list(selections.keys()))
    progress(1.0, desc="Listo!")
    yield outputs

def handle_chat(question, history, state: SessionState):
    # ── Guard: verificar login ──────────────────────────────────────────
    if not state.is_logged_in:
        return history + [{"role": "assistant", "content": "⛔ Debes iniciar sesión para usar el chat."}], "", state, state.cost_tracker.render(state.model)
    if not question.strip(): return history, "", state, state.cost_tracker.render(state.model)
    if state.df.empty:
        return history + [{"role": "assistant", "content": "⚠️ Ejecuta una busqueda primero."}], "", state, state.cost_tracker.render(state.model)
    # Activar el tracker para contabilizar el costo del chat (LLM + embedding de la consulta).
    from utils.cost_tracker import set_current_tracker
    set_current_tracker(state.cost_tracker)
    answer = rag_chat(question, state.session_id, state.user_id, state.chat_history,
                      state.model, df_fallback=state.df, stats=state.stats)
    state.chat_history += [{"role":"user","content":question}, {"role":"assistant","content":answer}]
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    # Guardar en Supabase chat_history
    try:
        from db.supabase_client import get_supabase_client, SCHEMA
        msgs = [
            {"user_id": state.user_id, "session_id": state.session_id, "role": "user", "content": question},
            {"user_id": state.user_id, "session_id": state.session_id, "role": "assistant", "content": answer},
        ]
        get_supabase_client().schema(SCHEMA).table("chat_history").insert(msgs).execute()
    except Exception as e:
        logger.warning(f"chat_history insert: {e}")

    return history, "", state, state.cost_tracker.render(state.model)

def handle_new_search(state: SessionState):
    """Reinicia datos de búsqueda manteniendo la sesión activa."""
    state.df = pd.DataFrame(); state.stats = {}; state.session_id = str(uuid.uuid4())
    state.errors = []; state.chat_history = []; state.discovery = {}
    state.cost_tracker.reset()
    new_html = get_social_selector_html(["youtube"])
    new_html += '<input type="hidden" id="social-hidden-real" value="youtube" />'
    return (f"🔆 Nueva sesion: `{state.session_id[:8]}`", new_html, state,
            state.cost_tracker.render(state.model), gr.update(visible=False))

def handle_model_change(model, state: SessionState):
    state.model = model
    badge = _build_badge(model)
    info = next((m for m in AVAILABLE_MODELS if m["id"] == model), None)
    p = info["pricing"] if info else ""
    return badge, f"`{model}` — {p}" if p else f"`{model}`", state

# Etiquetas del radio de centro de búsqueda (componentes NATIVOS de Gradio).
MAPS_CENTER_DEFAULT = "📍 Centro CDMX (por defecto)"
MAPS_CENTER_COORDS  = "✍️ Coordenadas manuales"
MAPS_CENTER_GPS     = "🧭 Mi ubicación (GPS)"


def handle_center_type_change(center_type: str):
    """Muestra/oculta el textbox de coordenadas y el botón GPS según el tipo de centro.
    Devuelve updates para (maps_location, btn_gps, maps_info)."""
    logger.info(f"📍 maps_center_type cambiado a: {center_type}")
    if center_type == MAPS_CENTER_COORDS:
        return (
            gr.update(visible=True, value=""),  # maps_location editable, vacío
            gr.update(visible=False),           # btn_gps
            "✍️ **Escribe las coordenadas** en formato `Latitud, Longitud` (ej. *19.4096, -99.1718*).",
        )
    elif center_type == MAPS_CENTER_GPS:
        return (
            gr.update(visible=True),            # maps_location (se llena con el GPS)
            gr.update(visible=True),            # btn_gps visible
            "🧭 Pulsa **Obtener mi ubicación GPS** (requiere abrir la app por `localhost` o `https`).",
        )
    # default (Centro CDMX)
    return (
        gr.update(visible=False, value="19.4326, -99.1332"),
        gr.update(visible=False),
        "📍 **Ubicación activa:** Centro de la Ciudad de México (`19.4326, -99.1332`).",
    )


def handle_location_change(location_text: str) -> str:
    """Valida en vivo las coordenadas escritas/detectadas y actualiza el aviso."""
    if not (location_text or "").strip():
        return "✍️ **Escribe las coordenadas** en formato `Latitud, Longitud` (ej. *19.4096, -99.1718*)."
    try:
        parts = location_text.split(",")
        if len(parts) == 2:
            lat = float(parts[0].strip())
            lng = float(parts[1].strip())
            return f"🟢 **Coordenadas activas:** `{lat}, {lng}`"
    except ValueError:
        pass
    return "⚠️ **Formato incorrecto:** usa `Latitud, Longitud` (ej. `19.4096, -99.1718`)."

# ── Wizard: KPIs + resumen + navegación entre pasos ──────────────────────
def render_kpis_and_summary(state: SessionState):
    """Construye el HTML de las KPI cards (Dashboard) y el Analysis Summary (Chat)
    a partir del estado ya poblado por la búsqueda. Se llama con .then() tras
    handle_search / handle_fetch_selected."""
    if state.df is None or state.df.empty:
        return "", ""
    kpi = get_kpi_cards_html(state.stats, state.df)
    summary = get_analysis_summary_html(state.stats, state.df, state.cost_tracker, state.model)
    return kpi, summary


def go_to_step(step: int):
    """Salta a un paso del wizard (tab nativo de Gradio)."""
    return gr.Tabs(selected=step)


# ─── UI ─────────────────────────────────────────────────────────────
SOCIAL_ALL = [d["value"] for d in SOCIAL_MEDIA_DEFS]
AUTO_MODE   = "⚡ Automático (Sí a todo)"
MANUAL_MODE = "🎯 Elegir posts/videos/apps"
HEADER_HTML = f"""
<div class="app-header">
    <h1>🕵️ ChismesitoGPT <span style="color:rgba(255,255,255,0.35);font-weight:300">v2</span></h1>
    <p>Descubre lo que dicen las redes sociales</p>
    {_build_provider_html()}
    <p style="font-size:11px;opacity:0.4;margin-top:6px">
       🚫 {len(PROHIBITED_MODELS)} modelos prohibidos filtrados
       | 🧠 Embed: {'✅ '+EMB_MODEL if EMB_MODEL else '❌ No detectado'}
    </p>
</div>"""

SOCIAL_JS = """
async () => {
    // Escribe al input HTML puro (NO Gradio) para evitar re-render
    function updateHidden(values) {
        const h = document.getElementById('social-hidden-real');
        if (!h) return;
        h.value = values.join(',') || 'youtube';
    }

    function bindChips(container) {
        if (container.dataset.bound === '1') return;
        container.dataset.bound = '1';

        container.addEventListener('click', (e) => {
            const chip = e.target.closest('.social-chip');
            if (!chip) return;
            chip.classList.toggle('selected');
            const selected = [...container.querySelectorAll('.social-chip.selected')]
                               .map(ch => ch.dataset.value);
            updateHidden(selected);
        });
    }

    function tryBind() {
        const c = document.getElementById('social-chips-container');
        if (c) { bindChips(c); return true; }
        return false;
    }

    // Primer intento inmediato
    if (!tryBind()) {
        // Observar el DOM hasta que aparezca el contenedor
        const observer = new MutationObserver(() => {
            if (tryBind()) observer.disconnect();
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    // Re-bind cuando Gradio re-renderiza el HTML (ej. Nueva Busqueda)
    document.addEventListener('gradio:render', () => {
        const c = document.getElementById('social-chips-container');
        if (c) { delete c.dataset.bound; bindChips(c); }
    });

    // ── Tarjetas de selección (Fase 1 del modo manual) ─────────────────────
    // Delegación global de clicks: robusta ante los re-render de Gradio (el
    // #selection-container se inyecta dinámicamente tras "Buscar posts/videos").
    function updateSelectionHidden() {
        const container = document.getElementById('selection-container');
        const h = document.getElementById('selection-hidden-real');
        if (!container || !h) return;
        const sel = {};
        container.querySelectorAll('.select-card.selected').forEach(card => {
            const p = card.dataset.platform, id = card.dataset.id;
            if (!p || !id) return;
            (sel[p] = sel[p] || []).push(id);
        });
        h.value = JSON.stringify(sel);
    }
    // Colorea los pines del mapa según lo seleccionado en el checklist de Maps.
    function updateMapSelection() {
        const gd = document.querySelector('#selection-map .js-plotly-plot');
        const container = document.getElementById('selection-container');
        if (!gd || !window.Plotly || !container || !gd.data || !gd.data[0]) return;
        const selectedNums = new Set();
        container.querySelectorAll('.select-card[data-platform="maps"].selected').forEach(card => {
            const n = parseInt(card.dataset.number || '0', 10);
            if (n) selectedNums.add(n);
        });
        const nums = gd.data[0].text || [];   // números de cada pin, en orden
        const colors = nums.map(t => selectedNums.has(parseInt(t, 10))
            ? '#0d9488' : 'rgba(148,163,184,0.45)');
        try { window.Plotly.restyle(gd, {'marker.color': [colors]}); } catch (e) {}
    }
    if (!window.__cardsBound) {
        window.__cardsBound = true;
        document.addEventListener('click', (e) => {
            if (e.target.closest('.sc-link')) return;   // los links no togglean
            const card = e.target.closest('.select-card');
            if (!card) return;
            card.classList.toggle('selected');
            updateSelectionHidden();
            if (card.dataset.platform === 'maps') updateMapSelection();
        });
    }

    // (El panel de Maps ahora usa componentes nativos de Gradio; el GPS se
    //  maneja con GPS_JS en el botón btn_gps.)
}
"""

# JS del botón GPS: lee la geolocalización del navegador y la escribe en el
# textbox nativo #maps-location-input (Gradio detecta el cambio vía 'input').
GPS_JS = """
() => {
    const el = document.querySelector('#maps-location-input textarea, #maps-location-input input');
    if (!navigator.geolocation) {
        alert('Tu navegador no soporta geolocalización. Escribe las coordenadas manualmente.');
        return;
    }
    navigator.geolocation.getCurrentPosition(
        (pos) => {
            const coords = pos.coords.latitude.toFixed(6) + ', ' + pos.coords.longitude.toFixed(6);
            if (el) { el.value = coords; el.dispatchEvent(new Event('input', { bubbles: true })); }
        },
        (err) => {
            let msg = err.message;
            if (err.code === 1) msg = 'Permiso denegado. Habilita la ubicación en tu navegador.';
            else if (err.code === 2) msg = 'Ubicación no disponible. Verifica tu conexión.';
            else if (err.code === 3) msg = 'Tiempo agotado. Revisa los Servicios de Ubicación.';
            alert('GPS: ' + msg + '\\n\\nNota: la geolocalización solo funciona si abres la app por ' +
                  'http://localhost:7860 o https:// (no por la IP de red).');
        },
        { enableHighAccuracy: false, timeout: 20000, maximumAge: 300000 }
    );
}
"""

with gr.Blocks(title="ChismesitoGPT v2") as demo:

    # ── Estado por usuario (gr.State — una instancia por conexión de browser) ──
    session_state = gr.State(lambda: SessionState())

    # ───────────────────────────────────────────────────────────────
    # PANEL DE LOGIN (visible por defecto)
    # ───────────────────────────────────────────────────────────────
    with gr.Column(visible=True, elem_id="login-panel") as login_panel:
        with gr.Column(elem_classes=["login-box"]):
            gr.HTML(f"""
<div style="text-align: center; margin-bottom: 24px;">
  {_get_logo_html()}
  <h1 style="margin: 0 0 4px; font-size: 26px; font-weight: 800; color: #f0f6fc;
             font-family: 'Inter', sans-serif; letter-spacing: -0.5px;">ChismesitoGPT</h1>
  <p style="margin: 0 0 12px; font-size: 13px; color: rgba(255,255,255,0.35);
            font-family: 'Inter', sans-serif;">Acceso restringido &mdash; inicia sesión</p>
  <p style="margin: 0; font-size: 12px; color: rgba(255,255,255,0.4);
            font-family: 'Inter', sans-serif;">Ingresa tus credenciales para continuar</p>
</div>""")
            login_email    = gr.Textbox(label="📧 Email", placeholder="tu@email.com",
                                        elem_id="login-email")
            login_password = gr.Textbox(label="🔒 Contraseña", type="password",
                                        elem_id="login-password")
            login_btn      = gr.Button("🔒 Iniciar Sesión 🔓", variant="primary", size="lg")
            login_status   = gr.Markdown("", elem_classes=["status-md"])

    # ───────────────────────────────────────────────────────────────
    # APP PRINCIPAL (oculto hasta login)
    # ───────────────────────────────────────────────────────────────
    with gr.Column(visible=False, elem_id="app-panel") as app_panel:

        # Header de usuario + botón logout
        user_header_html = gr.HTML("", elem_id="user-header")
        with gr.Row():
            gr.HTML(HEADER_HTML)
            logout_btn = gr.Button("🚪 Cerrar Sesión", variant="secondary",
                                   size="sm", scale=0, min_width=150)

        demo.load(fn=None, js=SOCIAL_JS)

        with gr.Tabs(elem_id="wizard-tabs") as tabs:

            # ═══════════════════════════════════════════════════════════════
            # PASO 1 — Configuración & Descubrimiento
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("①  Configuración & Descubrimiento", id=0):
                gr.Markdown("## Configurar búsqueda\n"
                            "Define los parámetros de recolección de inteligencia social para tu análisis.")

                prompt = gr.Textbox(label="¿Qué quieres investigar?",
                                    placeholder="Ej: Opiniones del iPhone 16 en Mexico", lines=2)

                # Calculadora de costos (accordion, como en Stitch)
                with gr.Accordion("💰 Calculadora de costos estimados", open=False):
                    cost_display = gr.Markdown(
                        "Ejecuta una búsqueda para ver el detalle de costos.",
                        elem_classes=["status-md"],
                    )

                # ── Modelo activo: selector (sincronizado con el del Chat) + banner ──
                model_selector = gr.Dropdown(
                    choices=MODEL_CHOICES or [("Gemini 2.5 Flash", "gemini-2.5-flash")],
                    value=DEFAULT_MODEL,
                    label="🧠 Selecciona un modelo",
                    info="Aplica tanto a la búsqueda/análisis como al chat.",
                    interactive=True,
                    min_width=260,
                    elem_classes=["model-selector-inline"],
                )
                model_info_banner = gr.HTML(_build_model_banner(DEFAULT_MODEL))

                # ── Selector de redes visual (chips SVG) ──
                # El input oculto es HTML puro (NO Gradio) para evitar re-render
                social_html = get_social_selector_html(["youtube"])
                social_html += '<input type="hidden" id="social-hidden-real" value="youtube" />'
                social_selector_html = gr.HTML(value=social_html)
                _social_dummy = gr.Textbox(value="unused", visible=False)

                with gr.Row():
                    max_comments = gr.Slider(minimum=1, maximum=50, value=10, step=1,
                                             label="Comentarios por plataforma", info="Menos = mas rapido")

                with gr.Accordion("📍 Configuración de Google Maps (Búsqueda Geográfica)", open=False):
                    maps_geo_toggle = gr.Checkbox(
                        label="Activar búsqueda geográfica por cercanía",
                        value=False,
                        info="Busca en un radio alrededor de una ubicación central en vez de búsqueda global."
                    )
                    with gr.Group(visible=False) as maps_geo_group:
                        maps_center_type = gr.Radio(
                            choices=[MAPS_CENTER_DEFAULT, MAPS_CENTER_COORDS, MAPS_CENTER_GPS],
                            value=MAPS_CENTER_DEFAULT,
                            label="Centro de búsqueda",
                        )
                        maps_location = gr.Textbox(
                            label="Coordenadas (Lat, Lng)",
                            value="19.4326, -99.1332",
                            placeholder="Ej: 19.4096, -99.1718",
                            visible=False,
                            elem_id="maps-location-input",
                        )
                        btn_gps = gr.Button("🧭 Obtener mi ubicación GPS", visible=False, size="sm")
                        maps_radius = gr.Slider(
                            label="Radio de búsqueda (metros)",
                            minimum=100, maximum=10000, value=2000, step=100,
                            info="Rango a la redonda desde el centro de búsqueda"
                        )
                        maps_info = gr.Markdown(
                            "📍 **Ubicación activa:** Centro de la Ciudad de México (`19.4326, -99.1332`).",
                            elem_classes=["status-md"],
                        )

                # ── Modo de extracción ──
                mode_selector = gr.Radio(
                    choices=[MANUAL_MODE, AUTO_MODE],
                    value=MANUAL_MODE,
                    label="¿Cómo quieres proceder?",
                    info="Manual: tú eliges de qué posts/videos/apps/lugares. Automático: extrae de los mejores resultados (Sí a todo).",
                )

                with gr.Row():
                    discover_btn   = gr.Button("🔎 Buscar posts/videos para elegir", variant="primary", size="lg", scale=4, visible=True)
                    search_btn     = gr.Button("🔍 Buscar y Analizar", variant="primary", size="lg", scale=4, visible=False)
                    new_search_btn = gr.Button("🔄 Nueva Búsqueda", variant="secondary", size="sm", scale=1)

                status = gr.Markdown("💡 Ingresa un prompt y selecciona redes sociales.",
                                     elem_classes=["status-md"])

                # ── Panel de selección (Fase 1 del modo manual) ──
                with gr.Column(visible=False) as selection_panel:
                    gr.Markdown("### 🎯 Contenido encontrado — elige de qué extraer comentarios")
                    selection_map = gr.Plot(label="Lugares encontrados (Google Maps)",
                                            visible=False, elem_id="selection-map")
                    selection_cards_html = gr.HTML("")
                    fetch_btn = gr.Button("✅ Obtener comentarios de lo seleccionado",
                                          variant="primary", size="lg")
                _selection_dummy = gr.Textbox(value="{}", visible=False)

            # ═══════════════════════════════════════════════════════════════
            # PASO 2 — Dashboard de Resultados
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("②  Dashboard", id=1):
                gr.Markdown("## Dashboard de resultados")
                kpi_cards_html = gr.HTML("")
                map_plot       = gr.Plot(label="Mapa de reseñas (Google Maps)")
                count_plot     = gr.Plot(label="Comentarios por plataforma")
                sentiment_plot = gr.Plot(label="Sentimiento")
                emotion_plot   = gr.Plot(label="Emociones")
                category_plot  = gr.Plot(label="Top categorías")

                with gr.Accordion("📋 Datos recolectados y descargas", open=False):
                    data_table    = gr.Dataframe(label="Comentarios recolectados",
                                                 interactive=False, wrap=True)
                    download_csv  = gr.File(label="⬇ CSV")
                    download_xlsx = gr.File(label="📥 Excel")

                with gr.Row():
                    back_to_config_btn = gr.Button("← Volver a configuración", variant="secondary", size="sm")
                    go_to_chat_btn     = gr.Button("Ir al Chat RAG →", variant="primary", size="lg")

            # ═══════════════════════════════════════════════════════════════
            # PASO 3 — Chat RAG con desglose de costos
            # ═══════════════════════════════════════════════════════════════
            with gr.TabItem("③  Chat RAG", id=2):
                gr.Markdown("## Deep-Dive: Chat con los datos")
                with gr.Row(equal_height=False):
                    # Panel lateral — Analysis Summary
                    with gr.Column(scale=2, min_width=280):
                        analysis_summary_html = gr.HTML(
                            '<div class="summary-card"><h3 class="sum-title">Resumen del análisis</h3>'
                            '<p style="color:rgba(255,255,255,0.4);font-size:13px;">'
                            'Ejecuta una búsqueda para ver el resumen y el desglose de costos.</p></div>'
                        )
                    # Panel principal — chat
                    with gr.Column(scale=5):
                        # Badge + selector de modelo (sincronizado con el del Paso 1)
                        with gr.Row(equal_height=True):
                            model_badge = gr.HTML(INITIAL_BADGE, elem_classes=["model-badge-wrap"])
                            model_selector_chat = gr.Dropdown(
                                choices=MODEL_CHOICES or [("Gemini 2.5 Flash", "gemini-2.5-flash")],
                                value=DEFAULT_MODEL,
                                label="🧠 Selecciona un modelo",
                                info="Cambia el modelo para la búsqueda y el chat.",
                                interactive=True,
                                scale=1,
                                min_width=260,
                                elem_classes=["model-selector-inline"],
                            )
                        model_status = gr.Markdown(f"`{DEFAULT_MODEL}`", visible=False)
                        chatbot = gr.Chatbot(label="", height=460)
                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="", placeholder="Pregunta sobre los resultados...",
                                scale=5, show_label=False, container=False
                            )
                            chat_btn = gr.Button("Enviar", variant="primary", scale=1, min_width=80)

                back_to_dashboard_btn = gr.Button("← Volver al Dashboard", variant="secondary", size="sm")

    # ── Events ───────────────────────────────────────────────────────────────

    # Wizard — navegación entre pasos (tabs nativos)
    back_to_config_btn.click(fn=lambda: go_to_step(0), inputs=None, outputs=[tabs])
    go_to_chat_btn.click(fn=lambda: go_to_step(2), inputs=None, outputs=[tabs])
    back_to_dashboard_btn.click(fn=lambda: go_to_step(1), inputs=None, outputs=[tabs])

    # Maps — panel geográfico con componentes nativos
    maps_geo_toggle.change(
        fn=lambda checked: gr.update(visible=bool(checked)),
        inputs=[maps_geo_toggle],
        outputs=[maps_geo_group],
    )
    maps_center_type.change(
        fn=handle_center_type_change,
        inputs=[maps_center_type],
        outputs=[maps_location, btn_gps, maps_info],
    )
    maps_location.change(
        fn=handle_location_change,
        inputs=[maps_location],
        outputs=[maps_info],
    )
    # GPS: lee la geolocalización del navegador y la escribe en maps_location.
    btn_gps.click(fn=None, js=GPS_JS)

    # Login
    login_btn.click(
        fn=handle_login,
        inputs=[login_email, login_password, session_state],
        outputs=[session_state, login_panel, app_panel, login_status, user_header_html]
    )
    login_password.submit(
        fn=handle_login,
        inputs=[login_email, login_password, session_state],
        outputs=[session_state, login_panel, app_panel, login_status, user_header_html]
    )

    # Logout
    logout_btn.click(
        fn=handle_logout,
        inputs=[session_state],
        outputs=[
            session_state, login_panel, app_panel,
            login_email, login_password, login_status, user_header_html
        ]
    )

    # Model change — sincroniza los dos selectores (Config ↔ Chat) + badge/banner.
    # Usamos .input() (solo interacción del usuario) para que actualizar un dropdown
    # de forma programática NO dispare el evento del otro (evita bucles infinitos).
    def _sync_model(model, state):
        badge  = _build_badge(model)
        info   = next((m for m in AVAILABLE_MODELS if m["id"] == model), None)
        p      = info["pricing"] if info else ""
        state.model = model
        status = f"`{model}` — {p}" if p else f"`{model}`"
        return gr.update(value=model), badge, status, _build_model_banner(model), state

    model_selector.input(
        fn=_sync_model,
        inputs=[model_selector, session_state],
        outputs=[model_selector_chat, model_badge, model_status, model_info_banner, session_state],
    )
    model_selector_chat.input(
        fn=_sync_model,
        inputs=[model_selector_chat, session_state],
        outputs=[model_selector, model_badge, model_status, model_info_banner, session_state],
    )

    # Modo de extracción: alterna botones y oculta el panel de selección
    def _handle_mode_change(mode):
        is_manual = (mode == MANUAL_MODE)
        return (
            gr.update(visible=not is_manual),   # search_btn
            gr.update(visible=is_manual),        # discover_btn
            gr.update(visible=False),            # selection_panel (se re-muestra al descubrir)
        )

    mode_selector.change(
        fn=_handle_mode_change,
        inputs=[mode_selector],
        outputs=[search_btn, discover_btn, selection_panel],
    )

    # Fase 1 (manual): descubrir posts/videos/apps
    discover_btn.click(
        fn=handle_discover,
        inputs=[
            prompt, _social_dummy, model_selector, max_comments,
            maps_geo_toggle, maps_location, maps_radius, session_state
        ],
        js="(prompt, dummy, model, maxc, geo, loc, rad, st) => "
           "[prompt, document.getElementById('social-hidden-real')?.value || 'youtube', "
           "model, maxc, geo, loc, rad, st]",
        outputs=[selection_cards_html, selection_panel, status, session_state, cost_display, selection_map],
    )

    # Fase 2 (manual): extraer comentarios solo de lo seleccionado
    fetch_btn.click(
        fn=handle_fetch_selected,
        inputs=[prompt, _selection_dummy, model_selector, max_comments, session_state],
        js="(prompt, dummy, model, maxc, st) => "
           "[prompt, document.getElementById('selection-hidden-real')?.value || '{}', "
           "model, maxc, st]",
        outputs=[
            sentiment_plot, emotion_plot, category_plot, count_plot, map_plot,
            status, data_table, download_csv, download_xlsx, session_state, cost_display
        ],
    ).then(
        fn=render_kpis_and_summary, inputs=[session_state],
        outputs=[kpi_cards_html, analysis_summary_html],
    ).then(
        fn=lambda: go_to_step(1), inputs=None, outputs=[tabs],
    ).then(
        fn=None,
        js="() => { [150, 500, 900].forEach(d => setTimeout(() => window.dispatchEvent(new Event('resize')), d)); }"
    )

    # Search
    search_btn.click(
        fn=handle_search,
        inputs=[
            prompt, _social_dummy, model_selector, max_comments,
            maps_geo_toggle, maps_location, maps_radius, session_state
        ],
        js="(prompt, dummy, model, maxc, geo, loc, rad, st) => "
           "[prompt, document.getElementById('social-hidden-real')?.value || 'youtube', "
           "model, maxc, geo, loc, rad, st]",
        outputs=[
            sentiment_plot, emotion_plot, category_plot, count_plot, map_plot,
            status, data_table, download_csv, download_xlsx, session_state, cost_display
        ]
    ).then(
        fn=render_kpis_and_summary, inputs=[session_state],
        outputs=[kpi_cards_html, analysis_summary_html],
    ).then(
        fn=lambda: go_to_step(1), inputs=None, outputs=[tabs],
    ).then(
        fn=None,
        js="() => { [150, 500, 900].forEach(d => setTimeout(() => window.dispatchEvent(new Event('resize')), d)); }"
    )

    # Nueva búsqueda — resetea datos y vuelve al Paso 1
    new_search_btn.click(
        fn=handle_new_search,
        inputs=[session_state],
        outputs=[status, social_selector_html, session_state, cost_display, selection_panel]
    ).then(
        fn=lambda: ("", ""), inputs=None,
        outputs=[kpi_cards_html, analysis_summary_html],
    ).then(
        fn=lambda: go_to_step(0), inputs=None, outputs=[tabs],
    )

    # Chat
    chat_btn.click(
        fn=handle_chat,
        inputs=[chat_input, chatbot, session_state],
        outputs=[chatbot, chat_input, session_state, cost_display]
    )
    chat_input.submit(
        fn=handle_chat,
        inputs=[chat_input, chatbot, session_state],
        outputs=[chatbot, chat_input, session_state, cost_display]
    )

if __name__ == "__main__":
    # HuggingFace Spaces define SPACE_ID; ahí NO matamos puertos (lo gestiona el runtime).
    ON_SPACES = bool(os.getenv("SPACE_ID"))
    PORT = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860")))

    # Matar proceso previo en el puerto (solo en local; evita error 10048 en Windows)
    if not ON_SPACES:
        if sys.platform == "win32":
            for line in os.popen(f"netstat -ano | findstr :{PORT}").read().splitlines():
                if "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    try:
                        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                        print(f"🔄 Proceso previo en puerto {PORT} (PID {pid}) terminado.")
                    except Exception:
                        pass
        else:
            os.system(f"fuser -k {PORT}/tcp 2>/dev/null")

    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        allowed_paths=[str(Path(os.path.dirname(__file__)) / "icons")],
        css=CUSTOM_CSS,
        js=SOCIAL_JS,     # ← Gradio 6.x: css y js van en launch(), no en Blocks()
    )

