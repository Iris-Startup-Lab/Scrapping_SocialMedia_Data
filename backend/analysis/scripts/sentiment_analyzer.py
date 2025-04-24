import os
import spacy
from transformers import pipeline, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from analysis.models import ResultadoAnalisisSentimiento
from scraping.models import Tweet
import tensorflow as tf

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Desactivar optimizaciones de oneDNN para evitar problemas de compatibilidad con TensorFlow y Hugging Face

try:
    print("Cargando modelo de Spacy...")
    # Cargar el modelo de Spacy en español
    nlp_spacy = spacy.load("es_core_news_md")  ## Modelo no tan pesado
except OSError:
    nlp_spacy = None
    print("Advertencia: No se pudo cargar el modelo de Spacy. Se utilizará Hugging Face.")

#sentiment_pipeline_hf = pipeline("sentiment-analysis", model="finiteautomata/beto-sentiment-analysis") # Modelo BETO para español
#sentiment_pipeline_hf = pipeline("sentiment-analysis", model="finiteautomata/beto-sentiment-analysis", framework="pt") # Modelo BETO para español

model_name = "finiteautomata/beto-sentiment-analysis"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
#model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True) # Cargar el modelo BETO para español
sentiment_pipeline_hf = pipeline("sentiment-analysis", model=model, tokenizer=model)
## Tensorflow no soporta el modelo BETO para español, por lo que se utiliza PyTorch


def analizar_sentimiento_hibrido(tweet):
    texto = tweet.full_text
    sentimiento = None
    score = None

    # Intentar análisis con Spacy
    if nlp_spacy:
        doc = nlp_spacy(texto)
        if doc.sentiment > 0.2:
            sentimiento = "positivo"
            score = doc.sentiment
        elif doc.sentiment < -0.2:
            sentimiento = "negativo"
            score = doc.sentiment
        else:
            print(f"Resultado no concluyente con Spacy para: '{texto[:50]}'. Intentando con Hugging Face.")
            # Si Spacy no es concluyente, intentar con Hugging Face
            resultado_hf = sentiment_pipeline_hf(texto)[0]
            sentimiento = resultado_hf['label']
            score = resultado_hf['score']
    else:
        resultado_hf = sentiment_pipeline_hf(texto)[0]
        sentimiento = resultado_hf['label']
        score = resultado_hf['score']

    # Guardar el resultado del análisis
    ResultadoAnalisisSentimiento.objects.create(tweet=tweet, texto_analizado=texto, sentimiento=sentimiento, score=score)
    print(f"Sentimiento analizado para: '{texto[:50]}': {sentimiento} (score: {score})")