# backend/analisis/models.py
from django.db import models
from scraping.models import Tweet


class AnalisisSentimiento(models.Model):
    texto = models.TextField()
    resultado = models.TextField()
    fecha_analisis = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Análisis de: {self.texto[:50]}"
    

### Ahora la clase para el análisis de sentimientos ###

class ResultadoAnalisisSentimiento(models.Model):
    tweet = models.ForeignKey(Tweet, 
                              on_delete=models.CASCADE, 
                              related_name='analisis_sentimientos')
    texto_analizado = models.TextField()
    sentimiento = models.CharField(max_length=50)  
    score = models.FloatField(null=True, blank=True)  
    fecha_analisis = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"Análisis de sentimientos para '{self.texto_analizado[:50]}': {self.sentimiento}"
