# -*- coding: utf-8 -*-
"""pipeline/analyzer.py — Pipeline de analisis de comentarios."""

import logging
import pandas as pd
import numpy as np
import time

from llm_manager import generate_categories_with_model
from tools.zero_shot_tool import classify_comment
from tools.sentiment_tool import analyze_sentiment_text
from tools.emotion_tool import analyze_emotion_text
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
    logger.info(f"Categorias: {candidate_labels}")

    if not candidate_labels:
        candidate_labels = ["General", "Opinion positiva", "Opinion negativa"]

    # Paso 2: Zero-shot clasifica cada comentario
    t_start_class = time.time()
    logger.info(f"Clasificando {len(comments)} comentarios con zero-shot...")
    df["category"] = [classify_comment(c, candidate_labels) for c in comments]
    t_end_class = time.time()

    # Paso 3: Sentimiento
    logger.info("Analizando sentimiento...")
    df["sentiment"] = [analyze_sentiment_text(c) for c in comments]

    # Paso 4: Emocion
    logger.info("Analizando emociones...")
    df["emotion"] = [analyze_emotion_text(c) for c in comments]

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
