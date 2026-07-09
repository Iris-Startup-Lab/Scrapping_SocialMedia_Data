# -*- coding: utf-8 -*-
"""ui/styles.py — CSS centralizado para ChismesitoGPT v2."""

import base64
import html
import json
from pathlib import Path

# ─── Leer SVGs y convertir a base64 data-URI ─────────────────────────────────
_ICONS_DIR = Path(__file__).parent.parent / "icons"

def _svg_to_data_uri(filename: str) -> str:
    """Convierte un SVG a data URI base64."""
    path = _ICONS_DIR / filename
    if not path.exists():
        return ""
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:image/svg+xml;base64,{b64}"

# Pre-cargar íconos
ICON_GEMINI     = _svg_to_data_uri("gemini.svg")
ICON_DEEPSEEK   = _svg_to_data_uri("deepseek.svg")
ICON_CLAUDE     = _svg_to_data_uri("claude.svg")
ICON_OPENAI     = _svg_to_data_uri("open-ai.svg")
ICON_DOWNLOAD_1 = _svg_to_data_uri("download-1.svg")
ICON_FILE_CSV   = _svg_to_data_uri("file-csv-solid-full.svg")
ICON_FILE_EXCEL = _svg_to_data_uri("file-excel-solid-full.svg")

# Mapeo provider → ícono
_PROVIDER_ICONS = {
    "google":    ICON_GEMINI,
    "deepseek":  ICON_DEEPSEEK,
    "anthropic": ICON_CLAUDE,
    "openai":    ICON_OPENAI,
}

# Mapeo provider → color de acento
_PROVIDER_COLORS = {
    "google":    "#4285F4",   # Azul Gemini
    "deepseek":  "#5B8DEF",   # Azul DeepSeek
    "anthropic": "#D97757",   # Naranja Claude
    "openai":    "#74AA9C",   # Verde OpenAI
}

# Mapeo provider → nombre legible
_PROVIDER_NAMES = {
    "google":    "Google Gemini",
    "deepseek":  "DeepSeek",
    "anthropic":  "Anthropic Claude",
    "openai":    "OpenAI",
}

# ─── Íconos de redes sociales ─────────────────────────────────────────────────
_SOCIAL_ICON_FILES = {
    "youtube":   "youtube.svg",
    "twitter":   "twitter-old.svg",
    "reddit":    "reddit.svg",
    "facebook":  "facebook.svg",
    "instagram": "instagram.svg",
    "tiktok":    "tiktok.svg",
    "maps":      "map-marker-1.svg",
    "playstore": "play-store.svg",
}

SOCIAL_ICONS: dict[str, str] = {
    key: _svg_to_data_uri(filename)
    for key, filename in _SOCIAL_ICON_FILES.items()
}

# Definición completa de cada red social
SOCIAL_MEDIA_DEFS: list[dict] = [
    {"value": "youtube",   "label": "YouTube"},
    {"value": "twitter",   "label": "X / Twitter"},
    {"value": "reddit",    "label": "Reddit"},
    {"value": "facebook",  "label": "Facebook"},
    {"value": "instagram", "label": "Instagram"},
    {"value": "tiktok",    "label": "TikTok"},
    {"value": "maps",      "label": "Google Maps"},
    {"value": "playstore", "label": "Play Store"},
]


def get_social_selector_html(selected: list[str]) -> str:
    """
    Genera un selector de redes sociales con chips visuales que muestran
    el SVG real de cada plataforma. Los chips marcados tienen clase 'selected'.
    El JS en app.py maneja los clicks y actualiza el estado Gradio.
    """
    chips = []
    for defn in SOCIAL_MEDIA_DEFS:
        val   = defn["value"]
        label = defn["label"]
        icon  = SOCIAL_ICONS.get(val, "")
        sel   = "selected" if val in selected else ""
        img_tag = (
            f'<img src="{icon}" alt="{label}" '
            f'style="width:22px;height:22px;filter:brightness(0) invert(1);opacity:0.85;" />'
            if icon else ""
        )
        chips.append(f"""
        <div class="social-chip {sel}" data-value="{val}" title="{label}">
            {img_tag}
            <span>{label}</span>
        </div>""")

    chips_html = "\n".join(chips)
    return f"""
<div style="margin-bottom:4px; font-size:11px; font-weight:600;
            letter-spacing:0.07em; color:rgba(255,255,255,0.4);
            text-transform:uppercase;">
    Redes sociales
</div>
<div id="social-chips-container" style="
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 4px 0;
">
    {chips_html}
</div>
"""


# ─── Tarjetas de selección (Fase 1 del flujo manual) ─────────────────────────

# Etiqueta legible por plataforma (para los encabezados de sección)
_PLATFORM_LABELS = {d["value"]: d["label"] for d in SOCIAL_MEDIA_DEFS}
_PLATFORM_LABELS["x_twitter"] = "X / Twitter"

# Sustantivo del item mostrado por plataforma
_ITEM_NOUN = {
    "youtube": "videos", "twitter": "tweets", "x_twitter": "tweets",
    "facebook": "posts", "instagram": "posts", "tiktok": "videos",
    "reddit": "posts", "playstore": "apps", "maps": "lugares",
}


def get_selection_cards_html(discovery: dict[str, list[dict]]) -> str:
    """
    Genera las tarjetas de selección agrupadas por plataforma (Fase 1).
    Cada tarjeta lleva data-platform + data-id y arranca 'selected'. El JS de
    app.py (bindCards) sincroniza la selección en el input oculto
    'selection-hidden-real' (JSON {platform:[id,...]}).
    """
    if not discovery or not any(discovery.values()):
        return (
            '<div class="status-md" style="margin-top:8px;">'
            '🔍 No se encontraron resultados para elegir. Prueba con otro término.'
            '</div>'
            '<input type="hidden" id="selection-hidden-real" value="{}" />'
        )

    sections = []
    default_selected: dict[str, list[str]] = {}

    # Maps va primero (para tener el mapa y su checklist juntos arriba).
    ordered = sorted(discovery.items(), key=lambda kv: 0 if kv[0] in ("maps", "google_maps") else 1)

    for platform, items in ordered:
        if not items:
            continue
        icon = SOCIAL_ICONS.get(platform, SOCIAL_ICONS.get("twitter", ""))
        label = _PLATFORM_LABELS.get(platform, platform.title())
        noun = _ITEM_NOUN.get(platform, "items")
        default_selected[platform] = [str(it.get("id", "")) for it in items if it.get("id")]

        cards = []
        for it in items:
            item_id = html.escape(str(it.get("id", "")), quote=True)
            title = html.escape(str(it.get("title", ""))[:140])
            author = html.escape(str(it.get("author", "")))
            stat = html.escape(str(it.get("stat", "")))
            url = html.escape(str(it.get("url", "")), quote=True)
            thumb = it.get("thumbnail") or ""
            if thumb:
                thumb_html = f'<img class="sc-thumb" src="{html.escape(thumb, quote=True)}" alt="" loading="lazy" />'
            else:
                thumb_html = (
                    f'<div class="sc-thumb sc-thumb-fallback">'
                    f'<img src="{icon}" alt="" style="width:26px;height:26px;filter:brightness(0) invert(1);opacity:0.6;" /></div>'
                )
            meta_bits = " · ".join([b for b in (author, stat) if b])
            link_html = (
                f'<a class="sc-link" href="{url}" target="_blank" rel="noopener" '
                f'onclick="event.stopPropagation()">Ver ↗</a>' if url else ""
            )
            num_attr = f' data-number="{int(it["number"])}"' if it.get("number") else ""
            cards.append(f"""
        <div class="select-card selected" data-platform="{platform}" data-id="{item_id}"{num_attr}>
            <div class="sc-check">✓</div>
            {thumb_html}
            <div class="sc-body">
                <div class="sc-title">{title or '(sin título)'}</div>
                <div class="sc-meta">{meta_bits}</div>
                {link_html}
            </div>
        </div>""")

        sections.append(f"""
    <div class="sc-section">
        <div class="sc-section-head">
            <img src="{icon}" alt="" style="width:20px;height:20px;filter:brightness(0) invert(1);opacity:0.85;" />
            <span>{label}</span>
            <span class="sc-count">{len(items)} {noun}</span>
        </div>
        <div class="sc-grid">{''.join(cards)}</div>
    </div>""")

    hidden_value = html.escape(json.dumps(default_selected), quote=True)
    return f"""
<div id="selection-container">
    <div class="sc-hint">
        Marca o desmarca los {', '.join(sorted({_ITEM_NOUN.get(p, 'items') for p in discovery}))}
        de los que quieres extraer comentarios. Todos vienen seleccionados por defecto (“Sí a todo”).
    </div>
    {''.join(sections)}
</div>
<input type="hidden" id="selection-hidden-real" value="{hidden_value}" />
"""


# ─── Wizard: stepper 1→2→3 ───────────────────────────────────────────────────
_WIZARD_STEPS = [
    ("Configuración & Descubrimiento", "Configuración"),
    ("Dashboard de Resultados", "Dashboard"),
    ("Chat RAG", "Chat"),
]


def get_stepper_html(active: int = 0) -> str:
    """
    Stepper visual estilo Stitch (1→2→3). `active` es el índice del paso actual.
    Los pasos anteriores se marcan como completados (✓). Cada nodo es clickeable:
    el JS de app.py (WIZARD_JS) traduce el click en un click al tab nativo oculto.
    """
    nodes = []
    for i, (long_label, _short) in enumerate(_WIZARD_STEPS):
        if i < active:
            state, mark = "done", "✓"
        elif i == active:
            state, mark = "active", str(i + 1)
        else:
            state, mark = "todo", str(i + 1)
        nodes.append(
            f'<div class="wz-step {state}" data-step="{i}" role="button" tabindex="0">'
            f'<div class="wz-dot">{mark}</div>'
            f'<span class="wz-label">{long_label}</span>'
            f'</div>'
        )
        if i < len(_WIZARD_STEPS) - 1:
            nodes.append(f'<div class="wz-line {"done" if i < active else ""}"></div>')
    return f'<div class="wz-stepper" id="wizard-stepper-inner">{"".join(nodes)}</div>'


# ─── KPI cards (fila superior del Dashboard) ─────────────────────────────────
def _net_sentiment(df) -> tuple:
    """Devuelve (net_pct, total_clasificados). net = (%pos - %neg). None si no hay datos."""
    if df is None or df.empty or "sentiment" not in df.columns:
        return None, 0
    s = df["sentiment"]
    pos = int((s == "Positivo").sum())
    neg = int((s == "Negativo").sum())
    tot = int(s.isin(["Positivo", "Negativo", "Neutral"]).sum())
    if tot == 0:
        return None, 0
    return round((pos - neg) / tot * 100), tot


def get_kpi_cards_html(stats: dict, df) -> str:
    """Fila de tarjetas KPI: top-3 plataformas por volumen + sentimiento neto."""
    tiles = []
    items = sorted((stats or {}).items(), key=lambda kv: kv[1], reverse=True)
    for platform, count in items[:3]:
        icon = SOCIAL_ICONS.get(platform, SOCIAL_ICONS.get("twitter", ""))
        label = _PLATFORM_LABELS.get(platform, platform.replace("_", " ").title())
        img = (f'<img src="{icon}" alt="" style="width:18px;height:18px;'
               f'filter:brightness(0) invert(1);opacity:0.85;" />' if icon else "")
        tiles.append(f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">{html.escape(label)}</span>
                <span class="kpi-icon">{img}</span>
            </div>
            <div class="kpi-value">{count:,}</div>
            <span class="kpi-sub">comentarios</span>
        </div>""")

    net, tot = _net_sentiment(df)
    if net is not None:
        sign = "+" if net >= 0 else ""
        arrow = "&#9650;" if net >= 0 else "&#9660;"
        cls = "accent" if net >= 0 else "accent-neg"
        tiles.append(f"""
        <div class="kpi-card {cls}">
            <div class="kpi-top">
                <span class="kpi-label">Sentimiento neto</span>
                <span class="kpi-icon">{arrow}</span>
            </div>
            <div class="kpi-value">{sign}{net}%</div>
            <span class="kpi-sub">{tot:,} clasificados</span>
        </div>""")

    if not tiles:
        return ""
    return f'<div class="kpi-row">{"".join(tiles)}</div>'


# ─── Analysis Summary (panel lateral del Paso 3 — Chat) ──────────────────────
def get_analysis_summary_html(stats: dict, df, cost_tracker=None, model: str = "") -> str:
    """Resumen del análisis estilo Stitch: totales, plataformas, sentimiento,
    desglose de costos por línea y temas clave."""
    stats = stats or {}
    total = sum(stats.values())

    pills = "".join(
        f'<span class="sum-pill">{html.escape(_PLATFORM_LABELS.get(p, p.title()))}</span>'
        for p in stats.keys()
    ) or '<span class="sum-pill off">Sin datos</span>'

    net, _tot = _net_sentiment(df)
    if net is None:
        sent_txt, sent_cls = "Sin datos", "neutral"
    elif net > 5:
        sent_txt, sent_cls = f"Positivo (+{net}%)", "pos"
    elif net < -5:
        sent_txt, sent_cls = f"Negativo ({net}%)", "neg"
    else:
        sent_txt, sent_cls = f"Neutral ({'+' if net >= 0 else ''}{net}%)", "neutral"

    llm   = cost_tracker.llm_cost_usd() if cost_tracker else 0.0
    emb   = cost_tracker.embed_cost    if cost_tracker else 0.0
    apify = cost_tracker.apify_cost    if cost_tracker else 0.0
    grand = llm + emb + apify

    topics = ""
    if df is not None and not df.empty and "category" in df.columns:
        top = df["category"].dropna().value_counts().head(4).index.tolist()
        topics = "".join(
            f'<span class="topic-chip">#{html.escape(str(t).replace(" ", "_"))}</span>'
            for t in top
        )
    topics = topics or '<span class="topic-chip off">—</span>'

    return f"""
<div class="summary-card">
    <h3 class="sum-title">Resumen del análisis</h3>

    <div class="sum-metric">
        <span class="sum-metric-label">Interacciones totales</span>
        <span class="sum-metric-value">{total:,}</span>
    </div>

    <div class="sum-block">
        <div class="sum-block-label">Plataformas encontradas</div>
        <div class="sum-pills">{pills}</div>
    </div>

    <div class="sum-block">
        <div class="sum-block-label">Sentimiento primario</div>
        <div class="sum-sentiment {sent_cls}">{sent_txt}</div>
    </div>

    <div class="sum-block">
        <div class="sum-block-label">Costos de sesión</div>
        <div class="sum-cost-row"><span>Búsqueda y extracción</span><span>${apify:.4f}</span></div>
        <div class="sum-cost-row"><span>Análisis LLM</span><span>${llm:.4f}</span></div>
        <div class="sum-cost-row"><span>Embeddings</span><span>${emb:.4f}</span></div>
        <div class="sum-cost-row sum-cost-total"><span>Costo total estimado</span><span>${grand:.4f}</span></div>
    </div>

    <div class="sum-block">
        <div class="sum-block-label">Temas clave</div>
        <div class="sum-topics">{topics}</div>
    </div>
</div>"""


def get_model_badge_html(model_id: str, model_label: str = "", provider: str = "") -> str:
    """
    Retorna HTML del badge del modelo activo para mostrar en el chat.
    Incluye el SVG del proveedor y el nombre del modelo.
    """
    icon_uri  = _PROVIDER_ICONS.get(provider, ICON_GEMINI)
    color     = _PROVIDER_COLORS.get(provider, "#4285F4")
    prov_name = _PROVIDER_NAMES.get(provider, provider.title())
    short_label = model_label.split("(")[0].strip() if model_label else model_id

    return f"""
<div id="model-badge" style="
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    backdrop-filter: blur(8px);
    width: fit-content;
    margin-bottom: 6px;
">
    <img src="{icon_uri}"
         style="width:28px; height:28px; filter: brightness(0) invert(1) opacity(0.85);"
         alt="{prov_name} icon" />
    <div style="line-height:1.2;">
        <div style="font-size:11px; color:rgba(255,255,255,0.45); font-family:'Inter',sans-serif; letter-spacing:0.05em; text-transform:uppercase;">
            Modelo activo
        </div>
        <div style="font-size:13px; color:{color}; font-weight:600; font-family:'Inter',sans-serif;">
            {short_label}
        </div>
    </div>
    <div style="
        width:7px; height:7px;
        border-radius:50%;
        background:{color};
        box-shadow: 0 0 6px {color};
        margin-left:4px;
    "></div>
</div>
"""


# ─── CSS principal ────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ───────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background: #0d1117 !important;
    color: #e6edf3 !important;
}

/* ── Header minimalista ─────────────────────────────────── */
.app-header {
    padding: 20px 0 12px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 20px;
}
.app-header h1 {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
    margin: 0 !important;
    letter-spacing: -0.3px;
}
.app-header p {
    font-size: 13px !important;
    color: rgba(255,255,255,0.4) !important;
    margin: 4px 0 0 !important;
}

/* ── Provider pills ─────────────────────────────────────── */
.provider-pills {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
}
.provider-pill {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.04);
}
.provider-pill.ok  { border-color: rgba(80,200,120,0.4); color: #50c878; }
.provider-pill.off { border-color: rgba(255,255,255,0.08); color: rgba(255,255,255,0.3); }

/* ── Input / Textarea ───────────────────────────────────── */
.gradio-container textarea,
.gradio-container input[type="text"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    transition: border-color 0.2s ease !important;
}
.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus {
    border-color: rgba(99,179,237,0.5) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.08) !important;
}

/* ── Dropdown ───────────────────────────────────────────── */
.gradio-container select,
.gradio-container .wrap {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: #e6edf3 !important;
    border-radius: 10px !important;
}

/* ── Botón primario ─────────────────────────────────────── */
.gradio-container button.primary,
.gradio-container .gr-button-primary {
    background: linear-gradient(135deg, #0d9488, #0891b2) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.2px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(13,148,136,0.3) !important;
}
.gradio-container button.primary:hover,
.gradio-container .gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(13,148,136,0.45) !important;
}

/* ── Botón secundario ───────────────────────────────────── */
.gradio-container button.secondary {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.7) !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}
.gradio-container button.secondary:hover {
    background: rgba(255,255,255,0.09) !important;
    border-color: rgba(255,255,255,0.2) !important;
}

/* ── Checkboxes redes sociales ──────────────────────────── */
.gradio-container .checkbox-group {
    gap: 8px !important;
}
.gradio-container .checkbox-group label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
    font-size: 13px !important;
    color: #e6edf3 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
.gradio-container .checkbox-group label:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(99,179,237,0.4) !important;
}
.gradio-container .checkbox-group input:checked + span {
    color: #63b3ed !important;
}

/* ── CHAT — zona principal ──────────────────────────────── */
.chat-section {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    margin: 16px 0 !important;
}

/* Burbujas del chatbot */
.gradio-container .message.user {
    background: rgba(13,148,136,0.18) !important;
    border: 1px solid rgba(13,148,136,0.25) !important;
    border-radius: 14px 14px 4px 14px !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container .message.bot {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 4px 14px 14px 14px !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Tabs ───────────────────────────────────────────────── */
.gradio-container .tab-nav {
    background: rgba(255,255,255,0.02) !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 4px 8px 0 !important;
    gap: 4px !important;
}
.gradio-container .tab-nav button {
    background: transparent !important;
    border: none !important;
    border-radius: 8px 8px 0 0 !important;
    color: rgba(255,255,255,0.45) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all 0.15s ease !important;
}
.gradio-container .tab-nav button.selected {
    color: #e6edf3 !important;
    background: rgba(255,255,255,0.07) !important;
    border-bottom: 2px solid #0d9488 !important;
}
.gradio-container .tabitem {
    background: rgba(255,255,255,0.015) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 16px !important;
}

/* ── Dataframe / tabla ──────────────────────────────────── */
.gradio-container table {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}
.gradio-container th {
    background: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.7) !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
.gradio-container td {
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    color: #e6edf3 !important;
}

/* ── Status / Markdown info ─────────────────────────────── */
.status-md {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 13px !important;
}

/* ── Labels ─────────────────────────────────────────────── */
.gradio-container label span {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: rgba(255,255,255,0.5) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }

/* ── Dashboard charts — evitar overflow y permitir scroll ─ */
.js-plotly-plot, .plot-container {
    width: 100% !important;
    max-width: 100% !important;
}
/* .plotly .main-svg {
    width: 100% !important;
} */
.tabitem > div {
    overflow-x: auto !important;
    max-width: 100% !important;
}

/* ── Social chips (selector de redes sociales) ──────────── */
.social-chip {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.15s ease;
    user-select: none;
}
.social-chip span {
    font-size: 13px;
    font-weight: 500;
    color: rgba(255,255,255,0.55);
    font-family: 'Inter', sans-serif;
    transition: color 0.15s ease;
    white-space: nowrap;
}
.social-chip img {
    transition: opacity 0.15s ease;
    opacity: 0.5;
}
.social-chip:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(99,179,237,0.35);
}
.social-chip:hover span {
    color: rgba(255,255,255,0.85);
}
.social-chip:hover img {
    opacity: 0.75;
}
.social-chip.selected {
    background: rgba(13,148,136,0.15);
    border-color: rgba(13,148,136,0.5);
    box-shadow: 0 0 0 1px rgba(13,148,136,0.2);
}
.social-chip.selected span {
    color: #5eead4;
    font-weight: 600;
}
.social-chip.selected img {
    opacity: 1;
    filter: brightness(0) saturate(100%) invert(82%) sepia(30%) saturate(400%) hue-rotate(130deg) !important;
}

/* ── Tarjetas de selección (Fase 1 del flujo manual) ────── */
#selection-container { margin-top: 8px; }
.sc-hint {
    font-size: 13px;
    color: rgba(255,255,255,0.55);
    background: rgba(13,148,136,0.08);
    border: 1px solid rgba(13,148,136,0.2);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 14px;
}
.sc-section { margin-bottom: 18px; }
.sc-section-head {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.sc-section-head span {
    font-size: 14px;
    font-weight: 600;
    color: #e6edf3;
}
.sc-section-head .sc-count {
    margin-left: auto;
    font-size: 11px;
    font-weight: 500;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.sc-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 10px;
}
.select-card {
    position: relative;
    display: flex;
    gap: 10px;
    padding: 10px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
    user-select: none;
}
.select-card:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(99,179,237,0.35);
}
.select-card .sc-check {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 20px;
    height: 20px;
    border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.2);
    background: rgba(0,0,0,0.3);
    color: transparent;
    font-size: 13px;
    font-weight: 700;
    line-height: 18px;
    text-align: center;
    transition: all 0.15s ease;
}
.select-card.selected {
    background: rgba(13,148,136,0.14);
    border-color: rgba(13,148,136,0.5);
    box-shadow: 0 0 0 1px rgba(13,148,136,0.2);
}
.select-card.selected .sc-check {
    background: linear-gradient(135deg,#0d9488,#0891b2);
    border-color: transparent;
    color: #fff;
}
.select-card .sc-thumb {
    width: 72px;
    height: 72px;
    border-radius: 8px;
    object-fit: cover;
    flex-shrink: 0;
    background: rgba(255,255,255,0.05);
}
.select-card .sc-thumb-fallback {
    display: flex;
    align-items: center;
    justify-content: center;
}
.select-card .sc-body {
    display: flex;
    flex-direction: column;
    gap: 3px;
    min-width: 0;
    padding-right: 22px;
}
.select-card .sc-title {
    font-size: 13px;
    font-weight: 600;
    color: #e6edf3;
    line-height: 1.25;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.select-card .sc-meta {
    font-size: 12px;
    color: rgba(255,255,255,0.5);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.select-card .sc-link {
    font-size: 12px;
    color: #5eead4;
    text-decoration: none;
    margin-top: auto;
}
.select-card .sc-link:hover { text-decoration: underline; }

/* ── Pantalla de Login ── */
.login-box {
    width: 100% !important;
    max-width: 420px !important;
    margin: 12vh auto !important; /* Centrado vertical y horizontal natural */
    padding: 40px 32px !important;
    background: rgba(13, 18, 30, 0.8) !important;
    border: 1px solid rgba(13,148,136,0.25) !important;
    border-radius: 24px !important;
    box-shadow: 0 20px 60px rgba(0,0,0,0.6) !important;
}
.login-box .status-md {
    margin-top: 15px !important;
}

/* ── Selector de modelo inline junto al badge ── */
.model-badge-wrap {
    display: flex;
    align-items: center;
    flex: 0 0 auto;
}
.model-selector-inline {
    display: flex;
    align-items: center;
}
.model-selector-inline select,
.model-selector-inline .wrap {
    height: 36px !important;
    min-height: 36px !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #e6edf3 !important;
    padding: 0 12px !important;
}

/* ── Wizard: tabs nativos de Gradio estilizados como pasos numerados ────── */
/* Navegación 100% nativa (confiable): los TabItems llevan ①②③ en su label. */
#wizard-tabs .tab-nav {
    display: flex !important;
    justify-content: center !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    padding: 6px 0 10px !important;
    margin-bottom: 16px !important;
}
#wizard-tabs .tab-nav button {
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    border-radius: 999px !important;
    color: rgba(255,255,255,0.5) !important;
    border: 1px solid transparent !important;
    background: rgba(255,255,255,0.04) !important;
    transition: all 0.18s ease !important;
}
#wizard-tabs .tab-nav button:hover {
    color: rgba(255,255,255,0.85) !important;
    background: rgba(255,255,255,0.08) !important;
}
#wizard-tabs .tab-nav button.selected {
    color: #ffffff !important;
    background: linear-gradient(135deg,#0d9488,#0891b2) !important;
    border-color: transparent !important;
    box-shadow: 0 2px 12px rgba(13,148,136,0.35) !important;
}
@media (max-width: 640px) {
    #wizard-tabs .tab-nav button { font-size: 13px !important; padding: 6px 12px !important; }
}

/* ── KPI cards (fila superior del Dashboard) ───────────────────────────── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 14px;
    margin-bottom: 18px;
}
.kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    backdrop-filter: blur(8px);
}
.kpi-card .kpi-top {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 4px;
}
.kpi-card .kpi-label {
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.05em; text-transform: uppercase;
    color: rgba(255,255,255,0.45);
}
.kpi-card .kpi-icon {
    width: 32px; height: 32px; border-radius: 9px;
    background: rgba(255,255,255,0.05);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; color: #34d399;
}
.kpi-card .kpi-value { font-size: 26px; font-weight: 700; color: #e6edf3; line-height: 1.1; }
.kpi-card .kpi-sub   { font-size: 11px; color: rgba(255,255,255,0.35); }
.kpi-card.accent      { border-color: rgba(13,148,136,0.5); box-shadow: 0 0 0 1px rgba(13,148,136,0.15); }
.kpi-card.accent .kpi-value     { color: #34d399; }
.kpi-card.accent-neg  { border-color: rgba(248,113,113,0.5); box-shadow: 0 0 0 1px rgba(248,113,113,0.15); }
.kpi-card.accent-neg .kpi-value { color: #f87171; }
.kpi-card.accent-neg .kpi-icon  { color: #f87171; }

/* ── Analysis Summary (panel del Chat, Paso 3) ─────────────────────────── */
.summary-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px;
}
.summary-card .sum-title {
    font-size: 18px; font-weight: 700; color: #e6edf3;
    margin: 0 0 16px;
}
.sum-metric {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 14px;
    display: flex; flex-direction: column; gap: 2px;
}
.sum-metric-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: rgba(255,255,255,0.4); }
.sum-metric-value { font-size: 24px; font-weight: 700; color: #5eead4; }
.sum-block { margin-bottom: 16px; }
.sum-block-label {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: rgba(255,255,255,0.4);
    margin-bottom: 8px;
}
.sum-pills, .sum-topics { display: flex; flex-wrap: wrap; gap: 6px; }
.sum-pill {
    font-size: 11px; font-weight: 600;
    padding: 4px 10px; border-radius: 8px;
    background: rgba(13,148,136,0.12);
    border: 1px solid rgba(13,148,136,0.3);
    color: #5eead4;
}
.sum-pill.off { background: rgba(255,255,255,0.04); border-color: rgba(255,255,255,0.1); color: rgba(255,255,255,0.4); }
.sum-sentiment { font-size: 15px; font-weight: 700; }
.sum-sentiment.pos     { color: #34d399; }
.sum-sentiment.neg     { color: #f87171; }
.sum-sentiment.neutral { color: #94a3b8; }
.sum-cost-row {
    display: flex; justify-content: space-between;
    font-size: 13px; padding: 4px 0;
    color: rgba(255,255,255,0.65);
}
.sum-cost-total {
    border-top: 1px solid rgba(255,255,255,0.1);
    margin-top: 6px; padding-top: 8px;
    font-weight: 700; color: #5eead4; font-size: 14px;
}
.topic-chip {
    font-size: 12px; font-weight: 500;
    padding: 4px 10px; border-radius: 8px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: #6cd3f7;
}
.topic-chip.off { color: rgba(255,255,255,0.35); }

/* ── Dataset y Descargas Premium ── */
.dataset-section {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin-top: 20px !important;
}

.dataset-section h3 {
    margin: 0 0 16px 0 !important;
}

.download-row {
    display: flex !important;
    gap: 12px !important;
    margin-bottom: 16px !important;
    flex-wrap: wrap !important;
}

.download-btn {
    flex: 1 !important;
    min-width: 180px !important;
    height: 42px !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

#download-csv-btn {
    background: rgba(34, 197, 94, 0.1) !important;
    border: 1px solid rgba(34, 197, 94, 0.3) !important;
    color: #4ade80 !important;
}
#download-csv-btn:hover {
    background: rgba(34, 197, 94, 0.2) !important;
    border-color: #22c55e !important;
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.15) !important;
}

#download-xlsx-btn {
    background: rgba(16, 185, 129, 0.1) !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    color: #34d399 !important;
}
#download-xlsx-btn:hover {
    background: rgba(16, 185, 129, 0.2) !important;
    border-color: #10b981 !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15) !important;
}

/* Redefinición Premium de tablas */
.gradio-container table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    width: 100% !important;
    background: rgba(10, 15, 30, 0.45) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

.gradio-container th {
    background: rgba(255, 255, 255, 0.05) !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    color: rgba(255, 255, 255, 0.9) !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    padding: 12px 16px !important;
    text-align: left !important;
}

.gradio-container tr {
    transition: background 0.15s ease !important;
}

.gradio-container tr:hover {
    background: rgba(255, 255, 255, 0.03) !important;
}

.gradio-container td {
    padding: 12px 16px !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
    color: rgba(230, 237, 243, 0.9) !important;
}

.gradio-container .table-wrap {
    border-radius: 12px !important;
    border: none !important;
}

/* ── Google Maps Highlight Card Premium ── */
.maps-highlight-card {
    background: rgba(249, 115, 22, 0.03) !important; /* Toque muy sutil del naranja de Maps */
    border: 1px solid rgba(249, 115, 22, 0.15) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin: 16px 0 !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
}

.maps-highlight-card:hover {
    border-color: rgba(249, 115, 22, 0.3) !important;
    box-shadow: 0 6px 24px rgba(249, 115, 22, 0.08) !important;
    background: rgba(249, 115, 22, 0.04) !important;
}

#maps-geo-toggle-checkbox {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 12px !important;
    width: 100% !important;
}

#maps-geo-toggle-checkbox:hover {
    border-color: rgba(249, 115, 22, 0.4) !important;
}

.maps-controls-group {
    background: rgba(0, 0, 0, 0.2) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-top: 10px !important;
}

.maps-status-info {
    background: rgba(249, 115, 22, 0.08) !important;
    border: 1px solid rgba(249, 115, 22, 0.2) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 13px !important;
    color: #ffd8a8 !important;
    margin-top: 14px !important;
}

/* ── Recolorización de Avatares en el Chat (Fondo Oscuro) ── */
.gradio-container .avatar-container img[src*="user-4.svg"],
.gradio-container .message-row img[src*="user-4.svg"],
.gradio-container .message img[src*="user-4.svg"] {
    filter: brightness(0) saturate(100%) invert(60%) sepia(80%) saturate(900%) hue-rotate(190deg) brightness(100%) contrast(100%) !important;
}

.gradio-container .avatar-container img[src*="gemini.svg"],
.gradio-container .message-row img[src*="gemini.svg"],
.gradio-container .message img[src*="gemini.svg"] {
    filter: brightness(0) saturate(100%) invert(53%) sepia(85%) saturate(1500%) hue-rotate(205deg) brightness(100%) contrast(100%) !important;
}

.gradio-container .avatar-container img[src*="deepseek.svg"],
.gradio-container .message-row img[src*="deepseek.svg"],
.gradio-container .message img[src*="deepseek.svg"] {
    filter: brightness(0) saturate(100%) invert(71%) sepia(74%) saturate(2300%) hue-rotate(185deg) brightness(100%) contrast(100%) !important;
}

.gradio-container .avatar-container img[src*="claude.svg"],
.gradio-container .message-row img[src*="claude.svg"],
.gradio-container .message img[src*="claude.svg"] {
    filter: brightness(0) saturate(100%) invert(60%) sepia(70%) saturate(1200%) hue-rotate(345deg) brightness(100%) contrast(100%) !important;
}

.gradio-container .avatar-container img[src*="open-ai.svg"],
.gradio-container .message-row img[src*="open-ai.svg"],
.gradio-container .message img[src*="open-ai.svg"] {
    filter: brightness(0) saturate(100%) invert(65%) sepia(45%) saturate(700%) hue-rotate(120deg) brightness(100%) contrast(100%) !important;
}

.gradio-container .avatar-container img[src*="robot-solid-full.svg"],
.gradio-container .message-row img[src*="robot-solid-full.svg"],
.gradio-container .message img[src*="robot-solid-full.svg"] {
    filter: brightness(0) saturate(100%) invert(71%) sepia(74%) saturate(2300%) hue-rotate(185deg) brightness(100%) contrast(100%) !important;
}
"""

