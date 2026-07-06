# -*- coding: utf-8 -*-
## Iris Startup Lab 
'''
<(*)
  ( >)
  /|
'''
### Fernando Dorantes Nieto
## Fecha de debuggeo: 2025-07-18

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from pysentimiento import create_analyzer

from shiny import  reactive

from transformers import pipeline
import torch # Puede ser necesario instalar pytorch


spacy_model = 'es_core_news_md' ### Esto puede cambiar a otro modelo más liviano si es necesario
spacy_nlp_sentiment = reactive.Value(None)
pysentimiento_analyzer_instance = reactive.Value(None)
pysentimiento_emotions_analyzer_instance = reactive.Value(None)    
summarizer_pipeline_instance = reactive.Value(None)
emotion_model = reactive.Value(None)
emotion_tokenizer = reactive.Value(None)
emotion_config = reactive.Value(None)

def _ensure_spacy_sentiment_model():
    if spacy_nlp_sentiment.get() is None:
        try:
            print('Iniciando el modelo de Spacy')
            nlp = spacy.load(spacy_model)
            if not nlp.has_pipe('spacytextblob'):
                nlp.add_pipe('spacytextblob')
            spacy_nlp_sentiment.set(nlp)
            print('Modelo Spacy cargado')
            return True 
        except Exception as e:
            print(f'Error al cargar el modelo de Spacy {e}')
            return False 
    return True #### Default Value 


def _ensure_pysentimiento_analyzer():
    if pysentimiento_analyzer_instance.get() is None:
        try: 
            print('Iniciando el modelo pysentimiento para sentimientos (valga la redundancia )')
            analyzer = create_analyzer(task='sentiment', lang='es')
            pysentimiento_analyzer_instance.set(analyzer)
            print('Modelo de sentimientos cargado')
            return True 
        except Exception as e: 
            print(f'Error al cargar el modelo de pysentimiento para sentimientos {e}')
    return True 



def _ensure_pysentimient_emotions_analyzer():
    if pysentimiento_emotions_analyzer_instance.get() is None:    
        try: 
            print('Iniciando el modelo pysentimiento para emociones X_X ')
            analyzer = create_analyzer(task='emotion', lang='es')
            pysentimiento_emotions_analyzer_instance.set(analyzer)
            print('Modelo de emociones cargado')
            return True 
        except Exception as e: 
            print(f'Error al cargar el modelo de pysentimiento para emociones {e}')
    return True 


def generate_emotions_analysis(text):
    text = str(text)
    if not _ensure_pysentimient_emotions_analyzer():
        return "Error: Modelo de emociones no disponible"
    analyzer = pysentimiento_emotions_analyzer_instance.get()
    if analyzer is None:
        return "Error interno modelo de emociones"
    emotion_map_es = {
        "joy": "Alegría", 
        "sadness": "Tristeza",
        "anger": "Enojo", 
        "fear": "Miedo", 
        "surprise": "Sorpresa", 
        "disgust": "Asco", 
        "neutral": "Neutral"
    }
    try:
        result = analyzer.predict(text)
        primary_emotion_en = result.output
        if primary_emotion_en=="others":
            primary_emotion_en="neutral"
        return emotion_map_es.get(primary_emotion_en, "Desconocida")
    except Exception as e:
        print('Error en el modelo de emociones {e}')
        return 'Análisis de emociones fallido'
    


def sentiment_based_on_emotions_analysis(text):
    text = str(text)
    emotions_available = ['Alegría', 'Tristeza', 'Enojo', 'Miedo', 'Sorpresa', 'Asco', 'Neutral', 'Desconocida']
    positive_emotions = ['Alegría', 'Sorpresa']
    negative_emotions = ['Tristeza', 'Enojo', 'Miedo', 'Asco']
    if text in emotions_available:
        if text in positive_emotions:
            return 'Positivo'
        elif text in negative_emotions:
            return 'Negativo'
        elif text=='Neutral':
            return 'Neutral'
    else:
        return 'Neutral'



def generate_sentiment_analysis(text):
    text= str(text) ## Hay que asegurarse que sea texto 
    #### Combinación entre spacy y pysentimiento  por si las dudas dudosas
    use_spacy = _ensure_spacy_sentiment_model() 
    nlp_model = spacy_nlp_sentiment.get() if use_spacy else None 
    if nlp_model:
        doc = nlp_model(text)
        polarity_spacy = doc._.blob.polarity
        if polarity_spacy > 0.1:
            return 'Positivo'  
        elif polarity_spacy < -0.1:
            return 'Negativo' 
        else:
            sentiment_spacy_neutral = 'Neutral' 
            use_pysentimiento = _ensure_pysentimiento_analyzer()
               
            analyzer_pysent = pysentimiento_analyzer_instance.get() if use_pysentimiento else None
            if analyzer_pysent: 
                try: 
                    result_pysentimiento = analyzer_pysent.predict(text)
                    pysent_sentiment_map = {'POS': 'Positivo', 'NEG': 'Negativo', 'NEU': 'Neutral'}
                    sentiment_pysent = pysent_sentiment_map.get(result_pysentimiento.output, 'Neutral')
                    if sentiment_pysent != 'Neutral':
                        return sentiment_pysent
                    else:
                        return sentiment_spacy_neutral
                except Exception as e: 
                 return sentiment_spacy_neutral
            else: 
                return sentiment_spacy_neutral
    else: 
        print('NLP no encontrado, ahora usemos pysentimiento')
        use_pysentimiento = _ensure_pysentimiento_analyzer()
        analyzer_pysent = pysentimiento_analyzer_instance.get() if use_pysentimiento else None

        if analyzer_pysent:
            try:
                result = analyzer_pysent.predict(text)
                pysent_sentiment_map = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutral"}
                return pysent_sentiment_map.get(result.output, "Neutral")
            except Exception as e:
                    print(f"Error en pysentimiento: {e}")
                    return "Error: Análisis de sentimiento (Pysentimiento) fallido"
        else: # Both Spacy and Pysentimiento are unavailable
            return "Error: Modelos de sentimiento no disponibles"        

#### Terminados los modelos de sentimiento y emociones 

####### Zona para el nuevo modelo de sentimientos y emociones basado en modelos pequeños de lenguaje

# Esta sería una alternativa a tu función _ensure_pysentimiento_analyzer
# Se puede inicializar de forma similar con reactive.Value() ## Zona de testeo
sentiment_pipeline = None

def _ensure_hf_sentiment_pipeline():
    global sentiment_pipeline
    if sentiment_pipeline is None:
        try:
            print("Iniciando pipeline de 'transformers' para sentimiento...")
            # Modelo popular y eficiente para análisis de sentimiento en varios idiomas, incluido español
            model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
            device = 0 if torch.cuda.is_available() else -1 # Usar GPU si está disponible
            sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, device=device)
            print("Pipeline de Transformers cargado.")
            return True
        except Exception as e:
            print(f"Error al cargar pipeline de Transformers: {e}")
            return False
    return True

def generate_sentiment_hf(text):
    """
    Función alternativa para generar sentimiento usando un pipeline de Transformers.
    """
    text = str(text)
    if not _ensure_hf_sentiment_pipeline() or sentiment_pipeline is None:
        return "Error: Modelo de Transformers no disponible"
    
    try:
        results = sentiment_pipeline(text)
        # El modelo 'nlptown' devuelve estrellas (1 a 5)
        # Podemos mapear esto a Positivo, Negativo, Neutral
        score = int(results[0]['label'].split()[0]) # Extrae el número de '5 stars'
        if score > 3:
            return "Positivo"
        elif score < 3:
            return "Negativo"
        else:
            return "Neutral"
    except Exception as e:
        print(f"Error en el análisis con Transformers: {e}")
        return "Análisis de sentimiento fallido"

