# UI.md — ChismesitoGPT v2: Diseño de Interfaz

> **Proposito:** Referencia de diseno para cambiar o redisenar la UI.
> **Framework:** Gradio 4+ gr.Blocks + CSS personalizado
> **Archivos clave:** ui/styles.py, app.py (linea 563+)

---

## 1. Sistema de Diseno

### Paleta de colores

| Token            | Valor                          | Uso                                     |
|------------------|--------------------------------|-----------------------------------------|
| bg-base          | #0d1117                        | Fondo global (casi negro, GitHub Dark)  |
| bg-surface       | rgba(255,255,255,0.04)         | Tarjetas, inputs, chips                 |
| bg-surface-hover | rgba(255,255,255,0.08)         | Hover sobre elementos                   |
| border-subtle    | rgba(255,255,255,0.10)         | Bordes generales                        |
| border-active    | rgba(13,148,136,0.5)           | Seleccionado / activo (teal)            |
| accent-primary   | #0d9488 a #0891b2             | Gradiente teal-cyan (boton primario)    |
| accent-teal      | #5eead4                        | Texto resaltado (links, chips activos)  |
| accent-blue      | #63b3ed                        | Focus de inputs                         |
| text-primary     | #e6edf3                        | Texto principal                         |
| text-secondary   | rgba(255,255,255,0.55)         | Texto de soporte                        |
| text-muted       | rgba(255,255,255,0.35)         | Labels, metadatos                       |

### Colores por proveedor de LLM

| Proveedor        | Color    |
|------------------|----------|
| Google Gemini    | #4285F4  |
| DeepSeek         | #5B8DEF  |
| Anthropic Claude | #D97757  |
| OpenAI           | #74AA9C  |

### Tipografia

- **Fuente:** Inter (Google Fonts) con fallback -apple-system, BlinkMacSystemFont, Segoe UI
- **Pesos:** 300 / 400 / 500 / 600 / 700
- **Tamanios:**
  - 11px — labels en caps, metadatos secundarios
  - 12px — badges, subtitulos de tarjeta
  - 13px — cuerpo de chips, botones, tarjetas
  - 14px — texto de inputs
  - 22px — h1 del header
  - 26px — h1 del login

### Espaciado y bordes

| Elemento             | Border-radius |
|----------------------|--------------|
| Inputs, chips        | 10px         |
| Tarjetas             | 12px         |
| Secciones / chat     | 16px         |
| Login box            | 24px         |

---

## 2. Componentes

### 2.1 Login Box
  max-width: 420px, centrado en 12vh vertical
  bg: rgba(13,18,30,0.8), border: teal 0.25 opacity, border-radius: 24px
  Contenido: Logo SVG 56px + h1 + subtitulo + Email + Password + Boton + Status

### 2.2 Banner Modelo Activo + Selector inline (gr.Row)
  [Logo | MODELO ACTIVO / Nombre modelo ●]  [Dropdown selector ▾ min-width:260px]
  Badge: width fit-content, borde sutil, punto glow con color del proveedor
  Dropdown: height 36px, fondo rgba(255,255,255,0.05), borde sutil

### 2.3 Social Chips
  [YouTube] [X/Twitter] [Reddit] [Facebook] [Instagram] [TikTok] [Maps] [PlayStore]
  Inactivo: fondo 0.04, texto 0.55 opacity, icono opacity 0.5
  Activo (.selected): fondo teal 0.15, borde teal, texto #5eead4, icono saturado teal
  Iconos: SVG reales convertidos a base64 data-URI desde icons/
  Transition: all 0.15s ease

### 2.4 Tarjetas de Seleccion (Fase 1 / Modo Manual)
  Grid: auto-fill minmax(260px, 1fr), gap 10px
  Cada tarjeta: Thumbnail 72x72 + Titulo (2 lineas clamp) + Autor + Ver link
  Seleccionada: bg teal 0.14, checkbox verde top-right
  Hover: fondo 0.08, borde azul 0.35

### 2.5 Chat RAG
  Burbuja usuario: rgba(13,148,136,0.18), radius 14px 14px 4px 14px
  Burbuja bot: rgba(255,255,255,0.05), radius 4px 14px 14px 14px
  Input inline con boton Enviar (gradiente teal)

### 2.6 Tabs Dashboard / Datos
  Tab inactivo: texto rgba(255,255,255,0.45)
  Tab activo: texto blanco + border-bottom 2px solid #0d9488

### 2.7 Boton Primario
  background: linear-gradient(135deg, #0d9488, #0891b2)
  border-radius: 10px
  box-shadow: 0 2px 12px rgba(13,148,136,0.3)
  hover: translateY(-1px) + sombra mas intensa (0.45)

---

## 3. Mapa de Pantallas

### A — Login (visible al cargar, oculto post-login)
  Fondo global #0d1117
  login-box centrado: Logo + ChismesitoGPT + subtitulo
  Input Email / Input Password
  Boton Iniciar sesion + Status

### B — Panel Principal (post-login)
  Header: Logo | ChismesitoGPT | Provider Pills (Gemini, DeepSeek, Claude)
  Accordion: Calculadora de costos (Markdown con breakdown)
  Row: Badge Modelo Activo | Dropdown selector inline
  Social Chips: las 8 redes
  Slider: Comentarios por plataforma (1-50, default 10)
  Accordion: Configuracion Google Maps
    Radio: Centro de busqueda (default / coordenadas / GPS)
    Input: Coordenadas Lat,Lng
    Boton: Obtener GPS
    Slider: Radio en metros (100-10000)
    Markdown: Ubicacion activa
  Radio: Manual (elegir posts) / Automatico (si a todo)
  Row: Boton Buscar posts | Boton Buscar y Analizar | Boton Nueva Busqueda
  Status Markdown

### C — Panel de Seleccion (Fase 1, modo Manual)
  Aparece debajo del status al hacer click en Buscar posts
  Hint instructivo (borde teal)
  Google Maps primero (si seleccionado): mapa Plotly + tarjetas
  Otras plataformas: grids de tarjetas seleccionables
  Boton: Obtener comentarios de lo seleccionado

### D — Chat RAG (siempre visible bajo busqueda)
  --- Chat con los datos ---
  Row: Badge Modelo + Dropdown selector inline
  Chatbot gr.Chatbot height=420px
  Row: Input pregunta (scale=5) | Boton Enviar (scale=1)

### E — Resultados (Tabs, aparecen tras analisis)
  --- Resultados ---
  Tab Dashboard:
    Mapa de resenas (solo Google Maps)
    Bar chart: comentarios por plataforma
    Bar chart: sentimiento (Positivo/Negativo/Neutral)
    Bar chart: emociones (Alegria, Tristeza, Enojo, etc.)
    Horizontal bar: Top categorias
  Tab Datos:
    Dataframe interactivo con todos los comentarios
    Descarga CSV
    Descarga Excel

---

## 4. Guia de Cambios de Diseno

### Cambiar paleta completa
  Editar valores hex/rgba en CUSTOM_CSS dentro de ui/styles.py (linea 266)

### Cambiar color de acento (teal por otro)
  Reemplazar #0d9488, #0891b2, #5eead4 y rgba(13,148,136,...) en styles.py

### Cambiar tipografia
  Editar la linea @import url(...) y font-family en el selector body, .gradio-container

### Cambiar badge del modelo
  Funcion get_model_badge_html() en styles.py linea 220

### Cambiar social chips
  Funcion get_social_selector_html() en styles.py linea 81
  CSS: clase .social-chip en styles.py linea 514

### Cambiar layout del chat
  Bloque # Chat RAG en app.py alrededor de linea 690

### Cambiar graficos
  ui/dashboard.py — funciones plot_sentiment(), plot_emotions(), plot_categories(), plot_map()

---

## 5. Archivos del UI

| Archivo         | Rol                                                              |
|-----------------|------------------------------------------------------------------|
| ui/styles.py    | CSS global + generadores HTML (badge, chips, tarjetas)           |
| ui/dashboard.py | Graficos Plotly del dashboard                                    |
| app.py          | Layout completo en gr.Blocks + todos los eventos de UI           |
| icons/          | SVGs de proveedores LLM y redes sociales (base64 en runtime)    |
