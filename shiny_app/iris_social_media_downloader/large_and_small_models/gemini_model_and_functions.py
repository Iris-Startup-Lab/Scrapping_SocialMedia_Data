# -*- coding: utf-8 -*-
## Iris Startup Lab 
'''
<(*)
  ( >)
  /|
'''

#-------------------------------------------------------------
######### Social Media Downloader Shiny App ######
######### VERSION 0.5 ######
######### Authors Fernando Dorantes Nieto
#-------------------------------------------------------------

#import google.generativeai as genai
import sys 
import os 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import google.generativeai as genai_old ### Antigua librería de Google, mantenimiento terminará el 31 de Agosto de 2025
## https://pypi.org/project/google-generativeai/   #### Librería antigua, repo
from google import genai 
from config import GEMINI_API_KEY
import pandas as pd 
from shiny import reactive 

##### Llamando los valores reactivos 
current_gemini_response = reactive.Value("Carga datos y luego haz una pregunta sobre ellos, o haz una pregunta general. Presiona Enter o el botón verde a la izquierda para activar el bot")
gemini_embeddings_model = reactive.Value(None)
gemini_model_instance = reactive.Value(None)


### Actualizar mes a mes y cambiar modelos a modelos más recientes
free_tier_limits = {
    'model': ['gemini-1.5-flash', 'gemini-2.0-flash', 
              'gemini-2.5-pro',
              'gemini-2.5-flash', 'gemini-2.0-flash-lite', 
              'gemma-3-4b-it'],
    'request_per_minute':[15, 15, 5, 10, 30, 30],
    'tokens_per_minute':[250000, 1000000, 250000, 250000,
                         1000000, 15000]
}

df_free_tier_limits = pd.DataFrame(free_tier_limits)

############################# NEW FUNCTIONS WITH NEW GEMINI API ######################################

#def _ensure_gemini_model(gemini_instance, gemini_response, selected_model= 'gemini-2.0-flash-lite'):
def _ensure_gemini_model():
    if gemini_model_instance.get() is None:
        if GEMINI_API_KEY:
            try:
                #genai.configure(api_key=GEMINI_API_KEY)
                #model = genai.GenerativeModel(selected_model)
                google_ai_client = genai.Client(api_key = GEMINI_API_KEY)
                model = google_ai_client.models
                gemini_model_instance.set(model)
            except Exception as e :
                current_gemini_response.set(f'Error: al iniciar el modelo {e} ')
        else: 
            current_gemini_response.set("Error: Clave de Gemini no configurada")
            return False 
    return gemini_model_instance.get() is not None 

#### Create the new function of topics generator 
def topics_generator(text, selected_model = 'gemini-2.0-flash-lite'):
    if not text:
        return "No hay texto para resumir"
    text = str(text)
    if not _ensure_gemini_model():
            return "Error: Modelo de Gemini no disponible"
    gemini_model = gemini_model_instance.get()
    max_gemini_input_len = 7000
    if len(text)>max_gemini_input_len:
        text_to_process = text[:max_gemini_input_len]
        print(f"Texto para resumen (Gemini) truncado a {max_gemini_input_len} caracteres")
    else: 
        text_to_process= text
    topics_prompt = (f'''
            Analiza el siguiente texto:
            "{text}"
            Tu tarea es extraer:
            1. Tópicos generales que engloben a todo el texto
            2. Genera 5 categorías relevantes y sencillas (máximo 4 palabras cada una) que resuman los temas principales o aspectos del texto
            IMPORTANTE: Formatea TODA tu respuesta EXCLUSIVAMENTE como una ÚNICA cadena de texto que represente una lista de Python.
            Esta lista debe contener strings. Cada string puede ser una funcionalidad o una categoría.
            Ejemplo de formato de respuesta esperado: ['funcionalidad A', 'funcionalidad B', 'categoría X', 'categoría Y', ..., 'categoría Z']
            No incluyas NINGÚN texto, explicación, ni markdown (como ```python ... ```) antes o después de esta lista. Solo la lista en formato de cadena.
            ''' 
    )
    try:
        print(f"Enviando a Gemini para topicos: {topics_prompt[:20]}...")
        #response = gemini_model.generate_content(topics_prompt)
        response = gemini_model.generate_content(model = selected_model, 
                                                 contents= topics_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error al generar tópicos con Gemini: {e}"
    


def _ensure_gemini_embeddings_model():
    if gemini_embeddings_model.get() is None:
        if GEMINI_API_KEY:
            try:
                print("Iniciando el Gemini 'Embedding'")    
                google_ai_client = genai.Client(api_key = GEMINI_API_KEY)
                model = google_ai_client.models
                gemini_embeddings_model.set("models/embedding-001") 
                print("Modelo Gemini 'Embedding' cargado")
                return True
            except Exception as e:
                print(f"Error al configurar Gemini {e}")
                return False    
        else:
            return False 
    return True 



############################# OLD FUNCTIONS ######################################
def _ensure_gemini_model_old():
    if gemini_model_instance.get() is None:
        if GEMINI_API_KEY:
            try:
                print('Iniciando el modelo de Gemini')
                genai_old.configure(api_key=GEMINI_API_KEY)
                model = genai_old.GenerativeModel('gemini-1.5-flash')
                gemini_model_instance.set(model)
                print('Modelo de Gemini inició correctamente')
            except Exception as e:
                print(f"Error al cargar el modelo de Gemini: {e}")
                current_gemini_response.set(f'Error: al iniciar el modelo {e} ')
        else:
            current_gemini_response.set("Error: Clave de Gemini no configurada")
            return False 
    return gemini_model_instance.get() is not None 

def topics_generator_old(text):
    if not text:
        return "No hay texto para resumir"
    text = str(text)
    if not _ensure_gemini_model():
        return "Error: Modelo de Gemini no disponible"
    gemini_model = gemini_model_instance.get()
    max_gemini_input_len = 7000
    if len(text) > max_gemini_input_len:
        text_to_process = text[:max_gemini_input_len]
        print(f"Texto para resumen (Gemini) truncado a {max_gemini_input_len} caracteres.")
    else: 
        text_to_process= text
    topics_prompt = (f'''
            Analiza el siguiente texto:
            "{text}"
            Tu tarea es extraer:
            1. Tópicos generales que engloben a todo el texto
            2. Genera 5 categorías relevantes y sencillas (máximo 4 palabras cada una) que resuman los temas principales o aspectos del texto
            IMPORTANTE: Formatea TODA tu respuesta EXCLUSIVAMENTE como una ÚNICA cadena de texto que represente una lista de Python.
            Esta lista debe contener strings. Cada string puede ser una funcionalidad o una categoría.
            Ejemplo de formato de respuesta esperado: ['funcionalidad A', 'funcionalidad B', 'categoría X', 'categoría Y', ..., 'categoría Z']
            No incluyas NINGÚN texto, explicación, ni markdown (como ```python ... ```) antes o después de esta lista. Solo la lista en formato de cadena.
            ''' 
    )
    try:
        print(f"Enviando a Gemini para topicos: {topics_prompt[:20]}...")
        response = gemini_model.generate_content(topics_prompt) #### Antigua librería
        return response.text.strip()
    except Exception as e:
        return f"Error al generar tópicos con Gemini: {e}"
    


def _ensure_gemini_embeddings_model_old():
    if gemini_embeddings_model.get() is None:
        if GEMINI_API_KEY:
            try:
                print("Iniciando el Gemini 'Embedding'")
                genai_old.configure(api_key=GEMINI_API_KEY)
                gemini_embeddings_model.set("models/embedding-001") 
                print("Modelo Gemini 'Embedding' cargado")
                return True
            except Exception as e:
                print(f"Error al configurar Gemini {e}")
                return False    
        else:
            return False 
    return True 


