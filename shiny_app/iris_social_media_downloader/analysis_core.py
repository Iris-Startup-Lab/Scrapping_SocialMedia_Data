# analysis_core.py
import logging
from typing import Any, List as TypingList
from functools import partial # Necesario si usas partial en las funciones de análisis

# Importar librerías de análisis
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from pysentimiento import create_analyzer
from transformers import pipeline
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate 
import pandas as pd # Necesario para pd.isna

logger = logging.getLogger(__name__)

# --- Funciones de Análisis de Texto (Globales, pero requieren acceso a modelos reactivos vía argumentos) ---
# Estas funciones se llamarán desde dentro del server, y recibirán las instancias de los modelos que serán reactive.Value.

def generate_sentiment_analysis_global(text: str, nlp_model_instance: Any, pysentimiento_analyzer_instance: Any, 
                                     _ensure_spacy_func: Any, _ensure_pysentimiento_func: Any) -> str:
    """Realiza análisis de sentimiento de texto, con fallback de Spacy a PySentimiento."""
    try:
        use_spacy = _ensure_spacy_func()
        nlp_model = nlp_model_instance.get() if use_spacy else None # Acceder al valor de Reactive

        if nlp_model:
            doc = nlp_model(text)
            polarity_spacy = doc._.blob.polarity
            if polarity_spacy > 0.1: return 'Positivo'
            elif polarity_spacy < -0.1: return 'Negativo'
            else: # Spacy es neutral, intenta PySentimiento
                use_pysentimiento = _ensure_pysentimiento_func()
                analyzer_pysent = pysentimiento_analyzer_instance.get() if use_pysentimiento else None # Acceder al valor
                if analyzer_pysent:
                    result_pysentimiento = analyzer_pysent.predict(text)
                    pysent_sentiment_map = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutral"}
                    return pysent_sentiment_map.get(result_pysentimiento.output, "Neutral")
                else: return "Neutral" # Pysentimiento no disponible
        else: 
            use_pysentimiento = _ensure_pysentimiento_func()
            analyzer_pysent = pysentimiento_analyzer_instance.get() if use_pysentimiento else None # Acceder al valor
            if analyzer_pysent:
                result = analyzer_pysent.predict(text)
                pysent_sentiment_map = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutral"}
                return pysent_sentiment_map.get(result.output, "Neutral")
            else: return "Error: Modelos de sentimiento no disponibles"
    except Exception as e:
        logger.error(f"Error en generate_sentiment_analysis (global scope): {e}", exc_info=True)
        return "Error: Análisis de sentimiento fallido"

def generate_emotions_analysis_global(text: str, emotions_analyzer_instance: Any, _ensure_emotions_func: Any) -> str:
    """Realiza análisis de emociones de texto usando PySentimiento."""
    try:
        if not _ensure_emotions_func(): return "Error: Modelo de emociones no disponible"
        analyzer = emotions_analyzer_instance.get() # Acceder al valor de Reactive
        if analyzer is None: return "Error interno modelo de emociones"
        emotion_map_es = {"joy": "Alegría", "sadness": "Tristeza", "anger": "Enojo", "miedo": "Miedo", "sorpresa": "Sorpresa", "disgust": "Asco", "neutral": "Neutral"}
        result = analyzer.predict(text)
        primary_emotion_en = result.output
        if primary_emotion_en=="others": primary_emotion_en="neutral"
        return emotion_map_es.get(primary_emotion_en, "Desconocida")
    except Exception as e:
        logger.error(f"Error en generate_emotions_analysis (global scope): {e}", exc_info=True)
        return "Error: Análisis de emociones fallido"

def topics_generator_global(text: str, gemini_model_instance: Any, _ensure_gemini_func: Any) -> str:
    """Genera tópicos clave a partir de un texto usando un modelo de Gemini."""
    try:
        if not _ensure_gemini_func(): return "Error: Modelo de Gemini no disponible"
        gemini_model = gemini_model_instance.get() # Acceder al valor de Reactive
        max_gemini_input_len = 7000
        text_to_process= text[:max_gemini_input_len] if len(text) > max_gemini_input_len else text
        topics_prompt = (f'''Analiza el siguiente texto: "{text_to_process}" Tu tarea es extraer: 1. Tópicos generales que engloben a todo el texto 2. Genera 5 categorías relevantes y sencillas (máximo 4 palabras cada una) que resuman los temas principales o aspectos del texto IMPORTANTE: Formatea TODA tu respuesta EXCLUSIVAMENTE como una ÚNICA cadena de texto que represente una lista de Python. Esta lista debe contener strings. Cada string puede ser una funcionalidad o una categoría. Ejemplo de formato de respuesta esperado: ['funcionalidad A', 'funcionalidad B', 'categoría X', 'categoría Y', ..., 'categoría Z'] No incluyas NINGÚN texto, explicación, ni markdown (como ```python ... ```) antes o después de esta lista. Solo la lista en formato de cadena.''')
        response = gemini_model.generate_content(topics_prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error en topics_generator (global scope): {e}", exc_info=True)
        return "Error al generar tópicos con Gemini"

def generate_zero_shot_classification_with_labels_global(text: str, candidate_labels: TypingList[str], topic_pipeline_instance: Any, _ensure_topics_func: Any) -> str:
    """Clasifica texto en tópicos predefinidos usando un pipeline zero-shot."""
    try:
        if not _ensure_topics_func(): return "Error: Pipeline de clasificación de temas no disponible"
        classifier = topic_pipeline_instance.get() # Acceder al valor de Reactive
        text_to_classify = str(text) if pd.isna(text) else text # Using pd.isna for consistency with pandas' NaN handling
        if not text_to_classify: return "Texto vacío o inválido"
        if classifier is None: return "Error interno modelo de tópicos" 
        if not candidate_labels or not isinstance(candidate_labels, list) or not all(isinstance(label, str) for label in candidate_labels):
            logger.error(f"Error: candidate_labels is not a valid list of strings: {candidate_labels}", exc_info=True)
            return "Error: Lista de etiquetas de clasificación no válida"               
        result = classifier(text_to_classify, candidate_labels)
        return result['labels'][0]
    except Exception as e:
        logger.error(f"Error en generate_zero_shot_classification_with_labels (global scope): {e}", exc_info=True)
        return 'No aplica'

def summary_generator_global(text: str, platform: str, summarizer_pipeline_instance: Any, gemini_model_instance: Any, 
                           _ensure_summarizer_func: Any, _ensure_gemini_func: Any) -> str:
    """Genera un resumen de texto usando BART o Gemini."""
    try:
        if not text: return "No hay texto para resumir"
        text = str(text)
        if platform=="wikipedia2":            
            if not _ensure_summarizer_func(): return "Error: Pipeline de resumen no disponible"
            summarizer = summarizer_pipeline_instance.get() # Acceder al valor de Reactive
            max_bart_input_len = 1024*3
            text_to_process = text[:max_bart_input_len] if len(text) > max_bart_input_len else text
            summary = summarizer(text_to_process, max_length=200, min_length=40, do_sample=False)[0]['summary_text']
            return f"Resumen (BART):\n{summary}"
        else: # Usar Gemini para otras plataformas
            if not _ensure_gemini_func(): return "Error: Modelo de Gemini no disponible"
            gemini_model = gemini_model_instance.get() # Acceder al valor de Reactive
            max_gemini_input_len = 1000
            text_to_process= text[:max_gemini_input_len] if len(text) > max_gemini_input_len else text
            summarization_prompt = (f"Por favor, resume el siguiente texto extraído de una plataforma de red social. Concéntrate en las ideas principales y el sentimiento general si es evidente. El texto es:\n\n---\n{text_to_process}\n---\n\nResumen conciso:")
            response = gemini_model.generate_content(summarization_prompt)
            return f"Resumen (Gemini):\n{response.text}"
    except Exception as e:
        logger.error(f"Error en summary_generator (global scope): {e}", exc_info=True)
        return f"Error al resumir: {e}"