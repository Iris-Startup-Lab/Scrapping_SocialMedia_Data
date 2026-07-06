# llm_models_setup.py
import logging
from typing import List
import pandas as pd
from shiny import reactive, ui

# Importar librerías de modelos
from google import genai  # Migrado de google.generativeai → google.genai
from transformers import pipeline

# Importar desde la configuración del proyecto
from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.gemini_model_instance = reactive.Value(None)
        self.gemini_embeddings_model = reactive.Value(None)
        self.summarizer_pipeline_instance = reactive.Value(None)
        self.topic_pipeline_instance = reactive.Value(None)
        self.current_gemini_response = reactive.Value("Carga datos y haz una pregunta.")

    DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-lite"

    def _ensure_gemini_model(self):
        if self.gemini_model_instance.get() is None:
            if not GEMINI_API_KEY:
                self.current_gemini_response.set("GEMINI_API_KEY no encontrada.")
                return False
            try:
                logger.info("Configurando el modelo Gemini (google.genai)...")
                client = genai.Client(api_key=GEMINI_API_KEY)
                self.gemini_model_instance.set(client)  # Guardar Client
                logger.info("Cliente Gemini (google.genai) configurado.")
                return True
            except Exception as e:
                error_msg = f"Error al configurar Gemini: {e}"
                logger.error(error_msg)
                self.current_gemini_response.set(error_msg)
                return False
        return True

    def _ensure_topics_pipeline(self):
        if self.topic_pipeline_instance.get() is None:
            try:
                logger.info('Cargando pipeline de tópicos')
                topicGenerator = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
                self.topic_pipeline_instance.set(topicGenerator)
                logger.info('Pipeline de tópicos cargado')
                return True
            except Exception as e:
                logger.error(f"Error al cargar pipeline de tópicos: {e}")
                return False
        return True

    def topics_generator(self, text: str) -> str:
        if not text:
            return "[]"
        if not self._ensure_gemini_model():
            return "['Error: Modelo Gemini no disponible']"
        
        client = self.gemini_model_instance.get()
        prompt = f'''
        Analiza el siguiente texto: "{text[:7000]}"
        Tu tarea es generar 5 categorías relevantes y sencillas (máximo 4 palabras cada una) que resuman los temas.
        Formatea tu respuesta EXCLUSIVAMENTE como una ÚNICA cadena de texto que represente una lista de Python.
        Ejemplo: ['funcionalidad A', 'categoría X', 'categoría Y']
        '''
        try:
            response = client.models.generate_content(model=self.DEFAULT_GEMINI_MODEL, contents=prompt)
            return response.text.strip()
        except Exception as e:
            return f"['Error al generar tópicos con Gemini: {e}']"

    def generate_zero_shot_classification_with_labels(self, text: str, candidate_labels: List[str]):
        if not self._ensure_topics_pipeline():
            return "Error: Pipeline no disponible"
        
        classifier = self.topic_pipeline_instance.get()
        text_to_classify = str(text) if pd.notna(text) else ""
        if not text_to_classify or not candidate_labels:
            return "No aplica"

        try:
            result = classifier(text_to_classify, candidate_labels)
            return result['labels'][0]
        except Exception as e:
            logger.error(f'Error al clasificar texto: {e}')
            return 'No aplica'
        
### 