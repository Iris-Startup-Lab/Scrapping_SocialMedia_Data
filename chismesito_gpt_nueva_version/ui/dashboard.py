# -*- coding: utf-8 -*-
"""ui/dashboard.py — Gráficos Plotly para el dashboard de ChismesitoGPT v2.

Gráficas:
  1. plot_platform_counts(stats)  → Donut: total de comentarios por plataforma
  2. plot_sentiment(df)           → Barras apiladas %: sentimiento por plataforma
  3. plot_emotions(df)            → Barras horizontales: emoción dominante global
  4. plot_categories(df)          → Treemap: distribución de categorías temáticas
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import logging

logger = logging.getLogger(__name__)
from config import MAPBOX_TOKEN

# ─── Tema compartido (dark, coherente con el CSS de la app) ──────────────────
_BG      = "#0d1117"
_PAPER   = "#161b22"
_GRID    = "rgba(255,255,255,0.06)"
_TEXT    = "#e6edf3"
_SUBTEXT = "rgba(230,237,243,0.45)"
_FONT    = "Inter, -apple-system, sans-serif"

_LAYOUT_BASE = dict(
    font=dict(family=_FONT, color=_TEXT, size=13),
    paper_bgcolor=_PAPER,
    plot_bgcolor=_BG,
    margin=dict(l=16, r=16, t=48, b=16),
    height=375,
    autosize=True,
    legend=dict(
        bgcolor="rgba(255,255,255,0.04)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(size=12),
    ),
)

# ─── Paletas ─────────────────────────────────────────────────────────────────
_SENTIMENT_COLORS = {
    "Positivo": "#34d399",   # verde esmeralda
    "Negativo": "#f87171",   # rojo suave
    "Neutral":  "#94a3b8",   # gris azulado
    "Error":    "#fbbf24",   # amarillo
}

_EMOTION_COLORS = [
    "#60a5fa",  # Alegría
    "#f472b6",  # Sorpresa
    "#a78bfa",  # Miedo
    "#fb923c",  # Enojo
    "#34d399",  # Asco
    "#94a3b8",  # Tristeza
    "#e2e8f0",  # Neutral / Desconocida
]

_PLATFORM_COLOR_MAP = {
    "youtube":     "#FF0000",      # Rojo
    "twitter":     "#1DA1F2",      # Azul claro
    "x_twitter":   "#1DA1F2",      # Azul claro
    "reddit":      "#FF4500",      # Naranja
    "facebook":    "#1877F2",      # Azul oscuro
    "instagram":   "#833AB4",      # Degradado (Rosa/Morado)
    "tiktok":      "#000000",      # Negro
    "google_maps": "#4285F4",      # Azul / "Verde" (as per hex spec)
    "maps":        "#4285F4",      # Azul / "Verde" (as per hex spec)
    "playstore":   "#00C851",      # Verde claro
}


# ─── 1. Donut — comentarios por plataforma ───────────────────────────────────
def plot_platform_counts(stats: dict) -> go.Figure | None:
    """
    Muestra la proporción de comentarios recolectados por plataforma.
    Si hay 3 o menos plataformas, usa un Donut Chart.
    Si hay más de 3, usa un Waffle Chart interactivo en Plotly.
    """
    if not stats:
        return None

    platforms = list(stats.keys())
    counts    = list(stats.values())
    total     = sum(counts)
    if total == 0:
        return None

    if len(platforms) <= 3:
        # Donut Chart
        colors = [_PLATFORM_COLOR_MAP.get(p.lower(), "#94a3b8") for p in platforms]
        fig = go.Figure(go.Pie(
            labels=platforms,
            values=counts,
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(size=13, family=_FONT),
            marker=dict(
                colors=colors,
                line=dict(color=_BG, width=2),
            ),
            hovertemplate="<b>%{label}</b><br>%{value} comentarios<br>%{percent}<extra></extra>",
        ))

        fig.update_layout(
            **_LAYOUT_BASE,
            title=dict(
                text=f"<b>{total:,}</b> comentarios recolectados",
                font=dict(size=15, family=_FONT),
                x=0.5,
            ),
            showlegend=True,
            annotations=[dict(
                text=f"<b>{total:,}</b><br><span style='font-size:11px;color:{_SUBTEXT}'>total</span>",
                x=0.5, y=0.5,
                font=dict(size=16, family=_FONT, color=_TEXT),
                showarrow=False,
            )],
        )
        return fig
    else:
        # Waffle Chart (Distribución en cuadrícula de 10x10)
        percentages = [c / total for c in counts]
        # Algoritmo de mayor residuo para que la suma de celdas sea exactamente 100
        tile_counts = [int(p * 100) for p in percentages]
        remainder = 100 - sum(tile_counts)
        if remainder > 0:
            decimals = [(p * 100 - int(p * 100), idx) for idx, p in enumerate(percentages)]
            decimals.sort(reverse=True, key=lambda x: x[0])
            for r in range(remainder):
                tile_counts[decimals[r][1]] += 1

        categories_list = []
        for idx, name in enumerate(platforms):
            categories_list.extend([name] * tile_counts[idx])

        x_coords = []
        y_coords = []
        names = []
        color_map = {name: _PLATFORM_COLOR_MAP.get(name.lower(), "#94a3b8") for name in platforms}
        
        for i in range(100):
            y = i // 10
            x = i % 10
            x_coords.append(x)
            y_coords.append(y)
            category_name = categories_list[i] if i < len(categories_list) else platforms[-1]
            names.append(category_name)

        fig = go.Figure()
        for idx, name in enumerate(platforms):
            p_x = [x_coords[i] for i in range(100) if names[i] == name]
            p_y = [y_coords[i] for i in range(100) if names[i] == name]
            if not p_x:
                continue
            fig.add_trace(go.Scatter(
                x=p_x,
                y=p_y,
                mode="markers",
                name=name,
                marker=dict(
                    symbol="square",
                    size=22,
                    color=color_map[name],
                    line=dict(color=_PAPER, width=2)
                ),
                hovertemplate=f"<b>{name}</b>: {tile_counts[idx]}% ({stats[name]:,} comentarios)<extra></extra>",
            ))

        fig.update_layout(
            **_LAYOUT_BASE,
            title=dict(
                text=f"<b>Distribución por Red Social</b> ({total:,} comentarios)",
                font=dict(size=15, family=_FONT),
                x=0.5,
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks="",
                showticklabels=False,
                range=[-0.5, 9.5],
                fixedrange=True,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks="",
                showticklabels=False,
                range=[-0.5, 9.5],
                scaleanchor="x",
                scaleratio=1,
                fixedrange=True,
            ),
        )
        fig.update_layout(margin=dict(l=24, r=24, t=64, b=24))
        return fig


# ─── 2. Barras apiladas % — sentimiento por plataforma ──────────────────────
def plot_sentiment(df: pd.DataFrame) -> go.Figure | None:
    """
    Barras horizontales apiladas en porcentaje (100%) por plataforma.
    Normalizado para poder comparar plataformas con distinto volumen.
    Solo muestra sentimientos que existen en los datos.
    """
    if df is None or df.empty:
        return None
    needed = {"sentiment", "social_media"}
    if not needed.issubset(df.columns):
        return None

    # Filtrar valores útiles
    valid = df[df["sentiment"].isin(["Positivo", "Negativo", "Neutral"])].copy()
    if valid.empty:
        return None

    cross = (
        pd.crosstab(valid["social_media"], valid["sentiment"], normalize="index") * 100
    ).round(1)

    fig = go.Figure()
    for sentiment in ["Positivo", "Neutral", "Negativo"]:
        if sentiment not in cross.columns:
            continue
        fig.add_trace(go.Bar(
            name=sentiment,
            y=cross.index.tolist(),
            x=cross[sentiment].tolist(),
            orientation="h",
            marker_color=_SENTIMENT_COLORS[sentiment],
            marker_line_width=0,
            hovertemplate=f"<b>%{{y}}</b> → {sentiment}: <b>%{{x:.1f}}%</b><extra></extra>",
            text=[f"{v:.0f}%" for v in cross[sentiment]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=11, color="white"),
        ))

    layout = dict(**_LAYOUT_BASE)
    layout["legend"] = dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="left", x=0,
        bgcolor="rgba(255,255,255,0.04)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(size=12),
    )
    fig.update_layout(
        **layout,
        barmode="stack",
        title=dict(text="Sentimiento por plataforma (%)", font=dict(size=14), x=0),
        xaxis=dict(
            title="Porcentaje",
            ticksuffix="%",
            range=[0, 100],
            gridcolor=_GRID,
            zeroline=False,
        ),
        yaxis=dict(title="", gridcolor=_GRID, automargin=True),
    )
    return fig


# ─── 3. Barras — emoción dominante global ───────────────────────────────────
def plot_emotions(df: pd.DataFrame) -> go.Figure | None:
    """
    Barras verticales de la distribución global de emociones.
    Ordenadas de mayor a menor para ver la emoción predominante al instante.
    (El crosstab por plataforma era ilegible con pocas filas)
    """
    if df is None or df.empty:
        return None
    if "emotion" not in df.columns:
        return None

    # Excluir valores sin sentido
    exclude = {"Error", "Desconocida", ""}
    series = df["emotion"].dropna()
    series = series[~series.isin(exclude)]
    if series.empty:
        return None

    counts = series.value_counts().reset_index()
    counts.columns = ["emotion", "count"]
    counts = counts.sort_values("count", ascending=True)  # ascendente para barh leer de mayor a menor

    colors = [_EMOTION_COLORS[i % len(_EMOTION_COLORS)] for i in range(len(counts))]

    fig = go.Figure(go.Bar(
        x=counts["count"],
        y=counts["emotion"],
        orientation="h",
        marker=dict(
            color=colors,
            line_width=0,
        ),
        text=counts["count"],
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate="<b>%{y}</b>: %{x} comentarios<extra></extra>",
    ))

    layout = {k: v for k, v in _LAYOUT_BASE.items() if k != "legend"}
    fig.update_layout(
        **layout,
        title=dict(text="Distribución de emociones", font=dict(size=14), x=0),
        xaxis=dict(title="Comentarios", gridcolor=_GRID, zeroline=False),
        yaxis=dict(title="", automargin=True),
        showlegend=False,
    )
    return fig


# ─── 4. Treemap — categorías temáticas ───────────────────────────────────────
def plot_categories(df: pd.DataFrame) -> go.Figure | None:
    """
    Treemap de categorías temáticas.
    Mucho más intuitivo que barras para comparar proporciones de categorías:
    el área visual del rectángulo = volumen de comentarios en esa categoría.
    """
    if df is None or df.empty:
        return None
    if "category" not in df.columns:
        return None

    counts = df["category"].dropna().value_counts().head(12).reset_index()
    counts.columns = ["category", "count"]
    if counts.empty:
        return None

    total = counts["count"].sum()
    counts["pct"] = (counts["count"] / total * 100).round(1)

    fig = go.Figure(go.Treemap(
        labels=counts["category"],
        parents=[""] * len(counts),
        values=counts["count"],
        textinfo="label+value+percent root",
        textfont=dict(size=13, family=_FONT),
        marker=dict(
            colorscale=[
                [0.0,  "#0f3460"],
                [0.35, "#0d9488"],
                [0.70, "#0891b2"],
                [1.0,  "#a5f3fc"],
            ],
            colorbar=dict(thickness=10, tickfont=dict(size=10)),
            showscale=False,
            line=dict(color=_BG, width=2),
        ),
        hovertemplate="<b>%{label}</b><br>%{value} comentarios<br>%{percentRoot:.1%} del total<extra></extra>",
    ))

    layout = {k: v for k, v in _LAYOUT_BASE.items() if k not in ("legend", "margin")}
    fig.update_layout(
        **layout,
        title=dict(text="Categorías temáticas", font=dict(size=14), x=0),
        margin=dict(l=8, r=8, t=48, b=8),
        showlegend=False,
    )
    return fig


def _extract_coords_from_url(url: str):
    """Extrae latitud y longitud de un formato de URL que contenga '&ll=lat,lng'."""
    import re
    if not url or not isinstance(url, str):
        return None, None
    m = re.search(r"[&?]ll=([-\d.]+),([-\d.]+)", url)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass
    return None, None


def plot_places_preview(places: list[dict]) -> go.Figure | None:
    """
    Mapa de vista previa (Fase 1 de Maps): un pin numerado por lugar encontrado,
    para que el usuario ubique visualmente los sitios antes de elegirlos en la
    lista de checkboxes. Los números coinciden con el prefijo del título.
    """
    if not places:
        return None
    pts = [p for p in places if p.get("lat") is not None and p.get("lng") is not None]
    if not pts:
        return None

    lats = [p["lat"] for p in pts]
    lngs = [p["lng"] for p in pts]
    nums = [str(p.get("number", i + 1)) for i, p in enumerate(pts)]
    names = [p.get("title", "") for p in pts]
    stats = [p.get("stat", "") for p in pts]

    fig = go.Figure(go.Scattermapbox(
        lat=lats, lon=lngs,
        mode="markers+text",
        marker=dict(size=26, color="#0d9488", opacity=0.9),
        text=nums,
        textfont=dict(size=12, color="white", family=_FONT),
        textposition="middle center",
        customdata=list(zip(names, stats)),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
    ))

    token_clean = MAPBOX_TOKEN.strip() if MAPBOX_TOKEN else ""
    use_mapbox = token_clean.startswith("pk.")
    fig.update_layout(
        font=dict(family=_FONT, color=_TEXT, size=13),
        paper_bgcolor=_PAPER, plot_bgcolor=_BG,
        mapbox_style="dark" if use_mapbox else "carto-darkmatter",
        mapbox_accesstoken=token_clean if use_mapbox else None,
        mapbox=dict(
            zoom=11,
            center=dict(lat=sum(lats) / len(lats), lon=sum(lngs) / len(lngs)),
        ),
        title=dict(text=f"📍 {len(pts)} lugares encontrados — elígelos abajo",
                   font=dict(size=14), x=0),
        margin=dict(l=0, r=0, t=48, b=0),
        height=420,
        showlegend=False,
    )
    return fig


def plot_map(df: pd.DataFrame) -> go.Figure | None:
    """Mapa: un punto por ubicacion, tamano = numero de comentarios en ese lugar."""
    if df is None or df.empty:
        return None

    map_df = df.copy()
    if "latitude" not in map_df.columns:
        map_df["latitude"] = None
    if "longitude" not in map_df.columns:
        map_df["longitude"] = None

    for idx, row in map_df.iterrows():
        lat = row.get("latitude")
        lng = row.get("longitude")
        if pd.isna(lat) or pd.isna(lng) or lat is None or lng is None:
            u_lat, u_lng = _extract_coords_from_url(str(row.get("url", "")))
            if u_lat is not None:
                map_df.at[idx, "latitude"] = u_lat
                map_df.at[idx, "longitude"] = u_lng

    map_df = map_df.dropna(subset=["latitude", "longitude"])
    if map_df.empty:
        return None

    try:
        map_df["latitude"] = pd.to_numeric(map_df["latitude"])
        map_df["longitude"] = pd.to_numeric(map_df["longitude"])

        # Agrupar por coordenadas (redondear a 4 decimales = ~11m de precision)
        map_df["round_lat"] = map_df["latitude"].round(4)
        map_df["round_lng"] = map_df["longitude"].round(4)
        grouped = map_df.groupby(["round_lat", "round_lng"]).agg(
            count=("comment", "count"),
            sample=("comment", lambda x: x.iloc[0][:80] + "..." if len(str(x.iloc[0])) > 80 else str(x.iloc[0])),
        ).reset_index()

        # Escalar tamano de marcador (min 12, max 40)
        max_count = grouped["count"].max()
        grouped["size"] = grouped["count"].apply(lambda c: 12 + (c / max(max_count, 1)) * 28)

        fig = go.Figure(go.Scattermapbox(
            lat=grouped["round_lat"],
            lon=grouped["round_lng"],
            mode="markers",
            marker=dict(size=grouped["size"], color="#0d9488", opacity=0.8),
            text=grouped["count"].apply(lambda c: f"{c} comentarios"),
            hoverinfo="text",
            hovertemplate="<b>%{text}</b><br><i>%{customdata}</i><extra></extra>",
            customdata=grouped["sample"],
        ))

        token_clean = MAPBOX_TOKEN.strip() if MAPBOX_TOKEN else ""
        use_mapbox = token_clean.startswith("pk.")
        center_lat = grouped["round_lat"].mean()
        center_lng = grouped["round_lng"].mean()
        fig.update_layout(
            font=dict(family="Inter, sans-serif", color="#e6edf3", size=13),
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
            mapbox_style="dark" if use_mapbox else "carto-darkmatter",
            mapbox_accesstoken=token_clean if use_mapbox else None,
            mapbox=dict(
                zoom=12,
                center=dict(lat=center_lat, lon=center_lng),
            ),
            title=dict(text=f"Mapa — {len(grouped)} ubicaciones, {grouped['count'].sum()} comentarios",
                       font=dict(size=14), x=0),
            margin=dict(l=0, r=0, t=48, b=0),
            height=500,
            showlegend=False,
        )
        return fig
    except Exception as e:
        logger.warning(f"Error generando mapa: {e}")
        return None

