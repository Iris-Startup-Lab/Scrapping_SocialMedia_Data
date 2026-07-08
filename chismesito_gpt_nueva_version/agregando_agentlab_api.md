# Agregar la API interna AGA2 (AgentLab / apibaz) como modelo LLM

**Fecha:** 8 de julio de 2026
**Pregunta:** ¿Se puede añadir la API interna `https://aga2-v1.ft.apibaz.com/` como
una opción de modelo LLM seleccionable (al estilo DeepSeek o Claude) dentro de
ChismesitoGPT?

**Respuesta corta:** Técnicamente **sí es posible**, pero **no encaja bien** como
un "modelo" intercambiable más, y el límite de **30 llamadas/día** lo vuelve
inviable como modelo por defecto o de uso general. Sirve, a lo mucho, como una
**opción experimental etiquetada** y con tope de uso. Abajo el detalle.

---

## 1. Qué es realmente esta API (y por qué importa)

Según su documentación (`/` y `/conversation-server.html`), **AGA2 no es una API de
chat/completions cruda** como las de Claude, DeepSeek, OpenAI o Gemini. Es una
**plataforma de orquestación de agentes**:

- Expone un **agente ya construido dentro de la plataforma**, con su propia
  persona, herramientas (tools) y memoria de conversación. No eliges "qué modelo
  LLM"; hablas con *ese agente*.
- La comunicación **no es una petición HTTP síncrona** que devuelve un texto.
  Es un flujo en dos pasos:
  1. **Handshake HTTP** a `/api/clientapikey` con un `apikey`
     (formato `sk-server-550e8400...`) para obtener un token.
  2. **Conexión WebSocket (Socket.IO)** al namespace `/agent`, con mensajería
     bidireccional por eventos.
- El flujo de mensajes es **por eventos y en streaming**:
  - Envías: `sendText(mensaje)` / `sendFile(nombre, base64)`.
  - Recibes eventos: `"agentReady"`, `"thinking"` (razonamiento),
    `"output"` `{type, content}` (respuesta final), `"fail"` (error).
- Respuesta con esquema propio: `executionId`, `inputTokenCount`,
  `outputTokenCount`, `inputCost`, `outputCost`, `outputs`, `isPartial`.
- Límites: máximo **2 archivos** por interacción; y en este demo,
  **30 llamadas/día** para el agente creado.

**Conclusión de esta sección:** es un *agente conversacional por WebSocket*, no un
*modelo de texto por REST*. Esa diferencia es la raíz de casi todos los contras.

---

## 2. Cómo integra modelos hoy ChismesitoGPT

El `llm_manager.py` tiene un contrato **muy simple y síncrono**:

- Cada proveedor implementa una función
  `_call_X(prompt, model, system_prompt) -> (texto, usage)` que hace **una
  petición y devuelve el texto completo** (sin streaming).
- `PROVIDER_CALL` mapea `provider -> función`; `STATIC_MODELS` cataloga cada
  modelo con `{id, provider, label, pricing, context, output}`.
- `get_llm_response()` elige el provider por el `id` del modelo, llama la función
  y registra el costo en `cost_tracker` a partir de `usage`.

Todos los proveedores actuales (Gemini SDK, OpenAI SDK, Anthropic REST, DeepSeek
SDK) siguen ese patrón **request → respuesta completa**. AGA2 rompe ese patrón
(WebSocket + eventos + streaming), así que habría que **adaptarlo**, no solo
"agregar una key".

---

## 3. ¿Es posible? Sí — así se vería el adaptador

Se podría escribir un `_call_agentlab(prompt, model, system_prompt) -> (texto, usage)`
que **oculte el WebSocket detrás del contrato síncrono**:

1. Handshake HTTP a `/api/clientapikey` → obtener token.
2. Abrir WebSocket Socket.IO al namespace `/agent` (nueva dependencia:
   `python-socketio[client]`).
3. Esperar `"agentReady"`, hacer `sendText(prompt)`.
4. **Acumular** los eventos `"output"` hasta que `isPartial == false`
   (convertir el streaming en un solo string).
5. Cerrar conexión; mapear `inputTokenCount/outputTokenCount` a `usage`.
6. Registrar el modelo como provider `"agentlab"` en `PROVIDER_CALL` y una
   entrada en `STATIC_MODELS`, p. ej.:
   `{"id": "aga2-v1", "provider": "agentlab", "label": "AGA2 Agent (demo · 30/día)", ...}`.

Es viable, pero **suma complejidad real**: manejo de event loop, timeouts,
reconexión, y el ciclo de vida de la conexión por cada llamada.

---

## 4. PROS

- **Es interna / propia.** No dependes de una API externa de pago; el cómputo y
  los datos se quedan en infraestructura de la empresa (relevante para datos
  sensibles).
- **Sin costo por token para nosotros** (dentro del demo): el gasto no golpea las
  keys de Gemini/Claude/DeepSeek.
- **Ya trae herramientas y memoria** integradas en la plataforma: si el agente
  AGA2 tiene tools útiles, podría hacer cosas que un modelo "pelón" no.
- **Devuelve métricas de tokens y costo** (`inputCost/outputCost`), fáciles de
  mapear al `cost_tracker` existente.
- **Soporta archivos** (hasta 2), útil si algún día se suben documentos.

---

## 5. CONTRAS

- **Es un agente, no un modelo intercambiable.** Tiene su propia persona y system
  prompt internos. ChismesitoGPT le pasa su propio `system_prompt`
  ("Eres ChismesitoGPT, analista de redes sociales…") y sus prompts de
  clasificación/RAG; el agente podría **ignorarlos o entrar en conflicto**. No se
  comporta como un modelo neutral al que le mandas cualquier instrucción.
- **Paradigma incompatible (WebSocket vs REST síncrono).** Todo el `llm_manager`
  asume request→respuesta. Envolver Socket.IO en una llamada bloqueante es posible
  pero frágil (event loop, timeouts, desconexiones) y agrega una dependencia nueva.
- **Límite de 30 llamadas/día (demo) — el bloqueante principal.** Ver §6.
- **Alcance de red para el despliegue.** El dominio `*.ft.apibaz.com` parece
  **interno**. Si se despliega en **Hugging Face Spaces** (plan de migración
  actual), es muy probable que el Space **no pueda alcanzar** un host interno →
  la opción fallaría en producción aunque funcione en local.
- **No aparece en el listado dinámico de modelos** (`list_available_models`);
  habría que hardcodearlo como entrada estática y mantenerlo a mano.
- **Sin streaming en la UI actual.** El adaptador tendría que *bufferear* toda la
  respuesta, perdiendo la ventaja de streaming del agente.
- **Un punto más de mantenimiento y falla** para una app que hoy tiene 4
  proveedores homogéneos y limpios.

---

## 6. Impacto real del límite de 30 llamadas/día

Cuántas llamadas al LLM hace la app hoy (verificado en el código):

| Acción | Llamadas al LLM | Nota |
|---|---|---|
| Generar categorías de una búsqueda (`analyzer.py`) | **1 por análisis** | usa una muestra, no por comentario |
| Sentimiento / emoción | **0** | corre en modelos HF locales, no cuenta |
| Embeddings | **0** contra este límite | usa Gemini embeddings (otra cuota) |
| Chat RAG (`rag.py`) | **1 por cada mensaje** del usuario | conversación multi-turno |

Las 30 llamadas son **compartidas entre TODOS los usuarios y TODOS los turnos de
chat del día**. Escenarios:

- Si se usa como **modelo del pipeline** (categorías): ~30 búsquedas/día en total.
  Ajustado para un demo con varios usuarios.
- Si se usa como **modelo del chat RAG**: una sola conversación de 15 mensajes se
  come **la mitad** de la cuota diaria. Dos usuarios platicando la agotan.

→ **No sirve como modelo por defecto ni de uso general.** A lo mucho, como opción
experimental para pruebas puntuales, con un **contador de cuota** que la
deshabilite al llegar al tope y evite dejar sin servicio al resto de la app.

---

## 7. Recomendación

**No agregarla como un "modelo" más en el dropdown principal** (al nivel de
Claude/DeepSeek). Las razones de fondo: es un *agente* con persona propia (no un
modelo neutral), usa un *paradigma WebSocket* ajeno a la arquitectura actual, y
sobre todo el *límite de 30/día* la hace inservible para uso normal.

Opciones ordenadas por sensatez:

1. **(Recomendada) Dejarla fuera del selector de modelos por ahora.** Retomarla
   cuando exista (a) una cuota real de producción y (b) confirmación de que el host
   es alcanzable desde el entorno de despliegue (HF Spaces / servidor final).
2. **Opción experimental acotada.** Si se quiere probar ya: añadir provider
   `"agentlab"` con el adaptador de §3, **etiquetado claramente**
   ("AGA2 · demo 30/día"), con **contador de cuota diario** y **restringido al
   chat RAG** (nunca al pipeline masivo). Útil para evaluar la calidad del agente,
   no para uso real.
3. **Integrarla como *feature* aparte, no como modelo.** Si el valor de AGA2 son
   sus *tools* y su memoria, tiene más sentido exponerla como un "modo Agente"
   separado del selector de LLM, en vez de forzarla al contrato de
   `get_llm_response`.

## 8. Antes de decidir — 2 cosas que hay que confirmar

- **Alcance de red:** ¿`aga2-v1.ft.apibaz.com` es accesible desde donde se
  desplegará la app (HF Spaces / servidor)? Si es solo intranet, la integración
  no funcionará en producción.
- **Comportamiento del agente:** ¿AGA2 es un asistente general al que le puedes
  mandar cualquier prompt, o está especializado en otra cosa? Si tiene una persona
  fija, chocará con los prompts de análisis y RAG de ChismesitoGPT.

---

*Documento generado como apoyo a la decisión. La integración es factible pero no
recomendada como modelo general mientras el límite sea de 30 llamadas/día y el
host sea interno.*
