import spacy
from transformers import pipeline
from analysis.models import ResultadoAnalisisSentimiento
from scraping.models import Tweet



try:
    nlp_spacy = spacy.load("es_core_news_sm")  # O el modelo de Spacy que prefieras
except OSError:
    nlp_spacy = None
    print("Advertencia: No se pudo cargar el modelo de Spacy. Se utilizará Hugging Face.")

sentiment_pipeline_hf = pipeline("sentiment-analysis", model="finiteautomata/beto-sentiment-analysis") # Modelo BETO para español



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