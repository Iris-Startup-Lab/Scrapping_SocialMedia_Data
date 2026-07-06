# -*- coding: utf-8 -*-
"""
utils/cost_tracker.py — Tracker de costos por sesión: LLM + embeddings + APIFY.

Cubre TODO el flujo de una sesión:
  - Muestra de comentarios y generación de categorías (LLM)   → tokens reales
  - Clasificación / re-embedding de comentarios (embeddings)  → tokens estimados
  - Cada corrida de APIFY (scraping)                          → pago por resultado
  - Cada llamada al chat RAG (LLM)                            → tokens reales

Uso normal (vía "tracker activo" por contexto, sin tocar cada firma):
    from utils.cost_tracker import CostTracker, set_current_tracker
    tracker = CostTracker()             # uno por SessionState
    set_current_tracker(tracker)        # al inicio de cada handler
    ...  # get_llm_response / embed_single / _run_actor registran solos
    print(tracker.render())
"""

import contextvars
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─── Precios LLM por modelo (USD por 1 millón de tokens) ────────────────────
# Fuente: precios verificados julio 2026 (input, output)
PRICING_TABLE: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite":     (0.10,  0.40),
    "gemini-2.5-flash":          (0.30,  2.50),
    "gemini-2.5-pro":            (1.25, 10.00),
    "gemini-3.1-flash-lite":     (0.25,  1.50),
    "gemini-3.5-flash":          (1.50,  9.00),
    "gemma-4-31b-it":            (0.07,  0.07),
    "deepseek-v4-flash":         (0.14,  0.28),
    "deepseek-v4-pro":           (0.435, 0.87),
    "claude-haiku-4-5-20251001": (0.80,  4.00),
    "claude-sonnet-4-5-20250929":(3.00, 15.00),
    "claude-sonnet-4-6":         (3.00, 15.00),
    "claude-opus-4-8":           (5.00, 25.00),
    "gpt-4o-mini":               (0.15,  0.60),
    "gpt-4o":                    (2.50, 10.00),
    "gpt-5-chat-latest":         (1.25, 10.00),
}
DEFAULT_PRICING = (0.10, 0.40)  # fallback para modelos no listados

# ─── Precios de embeddings (USD por 1 millón de tokens de entrada) ──────────
# Ajusta según tu plan de Gemini. text-embedding-004 es legacy/gratuito;
# los modelos gemini-embedding-* se cobran por token de entrada.
EMBEDDING_PRICING: dict[str, float] = {
    "text-embedding-004":            0.00,
    "models/gemini-embedding-001":   0.15,
    "models/gemini-embedding-2":     0.20,
}
DEFAULT_EMBEDDING_PRICE = 0.20

# ─── Precios de APIFY (USD por 1000 resultados) ─────────────────────────────
# Fuente: tools/apify_cost_actors_per_request.txt (pago por resultado).
# Se indexa por el actor_id tal como se usa en ACTOR_MAP (hash o nombre legible).
APIFY_PRICING_PER_1000: dict[str, float] = {
    # Facebook
    "Us34x9p7VgjCz99H6":  12.0,   # apify/facebook-search-scraper
    "KoJrdxJCTtpon81KY":   4.0,   # apify/facebook-posts-scraper
    "us5srxAYnsrkgUv2v":   2.0,   # apify/facebook-comments-scraper
    # X / Twitter
    "8CiMefkv2yLlD7vYl":   3.0,   # watcher.data/search-x-by-keywords
    "qhybbvlFivx7AP0Oh":   0.25,  # scraper_one/x-post-replies-scraper
    # Instagram
    "reGe1ST3OBgYZSsZJ":   2.3,   # apify/instagram-hashtag-scraper
    "SbK00X0JYCPblD2wp":   2.3,   # apify/instagram-comment-scraper
    # TikTok
    "f1ZeP0K58iwlqG2pY":   2.0,   # clockworks/tiktok-hashtag-scraper
    "BDec00yAmCm1QbMEI":   1.0,   # clockworks/tiktok-comments-scraper
    # Google Maps
    "compass/Google-Maps-Reviews-Scraper": 0.45,
    # Google Search (resolución de ligas de Play Store)
    "apify/google-search-scraper":     2.5,
    # Aliases por nombre legible (por si se pasa el nombre en vez del hash)
    "apify/facebook-search-scraper":   12.0,
    "apify/facebook-posts-scraper":     4.0,
    "apify/facebook-comments-scraper":  2.0,
    "watcher.data/search-x-by-keywords": 3.0,
    "scraper_one/x-post-replies-scraper": 0.25,
    "apify/instagram-hashtag-scraper":  2.3,
    "apify/instagram-comment-scraper":  2.3,
    "apify/instagram-post-scraper":     1.5,
    "clockworks/tiktok-hashtag-scraper": 2.0,
    "clockworks/tiktok-comments-scraper": 1.0,
}
DEFAULT_APIFY_PER_1000 = 1.0  # fallback conservador para actores no listados


def estimate_tokens(text: str) -> int:
    """Estimación rápida de tokens (~4 caracteres por token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class CostTracker:
    """Tracker de uso y costo por sesión de usuario (una instancia por SessionState)."""
    # LLM
    tokens_in:   int = 0
    tokens_out:  int = 0
    llm_calls:   int = 0
    by_model:    dict = field(default_factory=dict)   # model -> {"in","out","calls","cost"}
    # Embeddings
    embed_tokens: int = 0
    embed_calls:  int = 0
    embed_cost:   float = 0.0
    # APIFY
    apify_runs:    int = 0
    apify_results: int = 0
    apify_cost:    float = 0.0
    apify_actors:  list = field(default_factory=list)
    _last_model:   str = field(default="deepseek-v4-flash", repr=False)

    # ── LLM ──────────────────────────────────────────────────────────────
    def add_llm(self, tokens_in: int, tokens_out: int, model: str) -> None:
        """Registra una llamada al LLM con conteo de tokens (idealmente reales)."""
        tokens_in = int(tokens_in or 0)
        tokens_out = int(tokens_out or 0)
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.llm_calls += 1
        self._last_model = model

        price_in, price_out = PRICING_TABLE.get(model, DEFAULT_PRICING)
        cost = (tokens_in * price_in + tokens_out * price_out) / 1_000_000
        m = self.by_model.setdefault(model, {"in": 0, "out": 0, "calls": 0, "cost": 0.0})
        m["in"] += tokens_in
        m["out"] += tokens_out
        m["calls"] += 1
        m["cost"] += cost
        logger.debug(f"CostTracker LLM +{tokens_in}in/{tokens_out}out [{model}] ${cost:.6f}")

    def add_llm_from_text(self, prompt: str, response: str, model: str) -> None:
        """Registra estimando tokens desde los textos (fallback sin usage real)."""
        self.add_llm(estimate_tokens(prompt), estimate_tokens(response), model)

    # ── Embeddings ───────────────────────────────────────────────────────
    def add_embedding(self, tokens: int, model: str = "") -> None:
        """Registra una llamada de embedding (tokens estimados de entrada)."""
        tokens = int(tokens or 0)
        self.embed_tokens += tokens
        self.embed_calls += 1
        price = EMBEDDING_PRICING.get(model, DEFAULT_EMBEDDING_PRICE)
        self.embed_cost += tokens * price / 1_000_000
        logger.debug(f"CostTracker EMBED +{tokens}tok [{model}]")

    # ── APIFY ────────────────────────────────────────────────────────────
    def add_apify(self, actor_id: str = "", results: int = 0) -> None:
        """Registra un run de APIFY y su costo (pago por resultado, $/1000)."""
        results = int(results or 0)
        self.apify_runs += 1
        self.apify_results += results
        price_per_1000 = APIFY_PRICING_PER_1000.get(actor_id, DEFAULT_APIFY_PER_1000)
        self.apify_cost += results * price_per_1000 / 1000.0
        if actor_id and actor_id not in self.apify_actors:
            self.apify_actors.append(actor_id)
        logger.debug(f"CostTracker APIFY run #{self.apify_runs} [{actor_id}] {results} res "
                     f"${results * price_per_1000 / 1000.0:.6f}")

    # ── Costos ───────────────────────────────────────────────────────────
    def llm_cost_usd(self) -> float:
        """Suma del costo LLM usando el precio propio de cada modelo."""
        return sum(m["cost"] for m in self.by_model.values())

    def get_cost_usd(self, model: str | None = None) -> float:
        """Costo total: LLM + embeddings + APIFY."""
        return self.llm_cost_usd() + self.embed_cost + self.apify_cost

    def render(self, model: str | None = None) -> str:
        """String legible para la UI, con desglose de modelos, embeddings y APIFY."""
        llm_cost = self.llm_cost_usd()
        total = llm_cost + self.embed_cost + self.apify_cost

        lines = ["### 💰 Costos de la sesión"]

        # LLM por modelo
        if self.by_model:
            lines.append(f"**🧠 Modelos LLM** — {self.tokens_in:,} in / {self.tokens_out:,} out "
                         f"({self.llm_calls} llamada{'s' if self.llm_calls != 1 else ''}) · ${llm_cost:.4f}")
            for mname, m in self.by_model.items():
                lines.append(f"  • `{mname}`: {m['in']:,}/{m['out']:,} tok · {m['calls']} · ${m['cost']:.4f}")
        else:
            lines.append("**🧠 Modelos LLM** — sin uso · $0.0000")

        # Embeddings
        lines.append(f"**🔢 Embeddings** — {self.embed_tokens:,} tok "
                     f"({self.embed_calls} llamada{'s' if self.embed_calls != 1 else ''}) · ${self.embed_cost:.4f}")

        # APIFY
        actors_str = f" · {len(self.apify_actors)} actor{'es' if len(self.apify_actors) != 1 else ''}" \
                     if self.apify_actors else ""
        lines.append(f"**🕷️ APIFY** — {self.apify_runs} run{'s' if self.apify_runs != 1 else ''}, "
                     f"{self.apify_results:,} resultados{actors_str} · ${self.apify_cost:.4f}")

        # Total
        lines.append(f"**Σ Total estimado: ${total:.4f} USD**")
        return "\n\n".join(lines)

    def reset(self) -> None:
        """Reinicia todos los contadores (nueva sesión de búsqueda)."""
        self.tokens_in = self.tokens_out = self.llm_calls = 0
        self.by_model = {}
        self.embed_tokens = self.embed_calls = 0
        self.embed_cost = 0.0
        self.apify_runs = self.apify_results = 0
        self.apify_cost = 0.0
        self.apify_actors = []

    def to_dict(self) -> dict:
        """Serializa el tracker a dict (para logs o persistencia futura)."""
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "llm_calls": self.llm_calls,
            "by_model": self.by_model,
            "llm_cost_usd": round(self.llm_cost_usd(), 6),
            "embed_tokens": self.embed_tokens,
            "embed_calls": self.embed_calls,
            "embed_cost_usd": round(self.embed_cost, 6),
            "apify_runs": self.apify_runs,
            "apify_results": self.apify_results,
            "apify_cost_usd": round(self.apify_cost, 6),
            "apify_actors": self.apify_actors,
            "total_cost_usd": round(self.get_cost_usd(), 6),
        }


# ─── "Tracker activo" por contexto ──────────────────────────────────────────
# Permite que get_llm_response / embed_single / _run_actor registren costos sin
# tener que recibir el tracker como parámetro. Es context-local (seguro entre
# hilos: cada request de Gradio corre en su propio hilo/contexto).
_CURRENT: contextvars.ContextVar = contextvars.ContextVar("cost_tracker", default=None)


def set_current_tracker(tracker: "CostTracker | None") -> None:
    """Fija el tracker activo para el contexto actual (llamar al inicio del handler)."""
    _CURRENT.set(tracker)


def get_current_tracker() -> "CostTracker | None":
    return _CURRENT.get()


def record_llm(model: str, tokens_in: int, tokens_out: int) -> None:
    t = _CURRENT.get()
    if t is not None:
        t.add_llm(tokens_in, tokens_out, model)


def record_embedding(tokens: int, model: str = "") -> None:
    t = _CURRENT.get()
    if t is not None:
        t.add_embedding(tokens, model)


def record_apify(actor_id: str, results: int) -> None:
    t = _CURRENT.get()
    if t is not None:
        t.add_apify(actor_id, results)
