# -*- coding: utf-8 -*-
"""pipeline/analyzer.py — Pipeline de analisis de comentarios."""

import logging
import pandas as pd
import numpy as np
import time

from llm_manager import generate_categories_with_model
from tools.zero_shot_tool import classify_comments_batch
from tools.emotion_tool import analyze_emotions_batch
from tools.embeddings_tool import embed_single

logger = logging.getLogger(__name__)


def analyze_comments(df: pd.DataFrame, sample_pct: float = 0.20, model: str = "gemini-2.5-flash", metrics: dict | None = None) -> pd.DataFrame:
    """
    Enriquecer un DataFrame de comentarios con:
      - category (zero-shot sobre categorias generadas por LLM)
      - sentiment (PySentimiento)
      - emotion (PySentimiento)
      - embedding (Gemini 768d)

    Args:
        df: DataFrame con columna 'comment'
        sample_pct: % de comentarios a usar para generar categorias
        model: Modelo LLM para generar categorias
        metrics: Diccionario para almacenar métricas de tiempo de ejecución
    """
    if df.empty or "comment" not in df.columns:
        logger.warning("DataFrame vacio o sin columna 'comment'")
        return df

    df = df.copy()
    comments = df["comment"].fillna("").astype(str).tolist()

    # Paso 1: LLM genera categorias (muestra de 20%)
    sample_size = max(5, int(len(comments) * sample_pct))
    sample_texts = comments[:sample_size]
    combined_sample = "\n---\n".join(sample_texts)

    logger.info(f"Generando categorias con muestra de {sample_size} comentarios (model={model})...")
    candidate_labels = generate_categories_with_model(combined_sample, model=model)

    if not candidate_labels:
        # El LLM no devolvió categorías válidas (ya quedó logueado el porqué en llm_manager).
        logger.warning("Categorías: usando fallback genérico ['General','Opinión positiva','Opinión negativa'] "
                       "— la categorización saldrá poco específica en esta corrida.")
        candidate_labels = ["General", "Opinión positiva", "Opinión negativa"]
    else:
        logger.info(f"Categorias: {candidate_labels}")

    # Paso 2: Zero-shot clasifica cada comentario
    t_start_class = time.time()
    logger.info(f"Clasificando {len(comments)} comentarios con zero-shot (batch)...")
    df["category"] = classify_comments_batch(comments, candidate_labels)
    t_end_class = time.time()

    # Visibilidad: cuántos comentarios quedaron sin categoría real (falla del zero-shot).
    n_err = int((df["category"] == "Error").sum())
    if n_err:
        logger.warning(f"Zero-shot: {n_err}/{len(df)} comentarios quedaron con category='Error' "
                       f"(falló la clasificación para esos textos).")

    # Paso 3+4: Emoción (un solo modelo, batch) → sentimiento derivado de la emoción
    logger.info("Analizando emoción y derivando sentimiento (batch)...")
    emo_sent = analyze_emotions_batch(comments)
    df["emotion"]   = [e for (e, _s) in emo_sent]
    df["sentiment"] = [s for (_e, s) in emo_sent]

    # Paso 5: Embeddings (Gemini 768d)
    t_start_embed = time.time()
    logger.info("Generando embeddings...")
    df["embedding"] = [embed_single(c) for c in comments]
    t_end_embed = time.time()

    if metrics is not None:
        metrics["classification_time"] = t_end_class - t_start_class
        metrics["embedding_time"] = t_end_embed - t_start_embed

    logger.info(f"Analisis completado: {len(df)} comentarios enriquecidos")
    return df
