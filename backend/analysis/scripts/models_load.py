#### Comenzando el script para la carga de modelos 
import os 
import sys
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from spacytextblob.spacytextblob import SpacyTextBlob
from googleapiclient.discovery import build
import spacy 

from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from tqdm import tqdm

### Cargando el modelo para la clasificación de sentimientos
analyzer = SentimentIntensityAnalyzer()

tokenizer = T5Tokenizer.from_pretrained("t5-base")
modelSummary = T5ForConditionalGeneration.from_pretrained("t5-base")


sentiment_classifier = pipeline('sentiment-analysis',
                                model="nlptown/bert-base-multilingual-uncased-sentiment")


nlp = spacy.load("es_core_news_lg")

# Inicializar el analizador de VADER
analyzer = SentimentIntensityAnalyzer()

### Cargando el modelo para la clasificación de emociones 







