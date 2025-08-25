# -*- coding: utf-8 -*-
## Iris Startup Lab 
'''
<(*)
  ( >)
  /|
'''
import sys 
import os 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from shiny import reactive 

from sentiments_and_emotions_classifier.emotions_and_sentiments import summarizer_pipeline_instance



def _ensure_summarizer_pipeline():
    if summarizer_pipeline_instance.get() is None:
        try:
            print('Cargando pipeline de resumen')
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summarizer_pipeline_instance.set(summarizer)
            print('Pipeline de resumen cargado')
            return True
        except Exception as e : 
            print(f"Error al cargar el pipeline de resumen {e}")
            return False 
    return True 
