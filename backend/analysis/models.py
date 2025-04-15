# backend/analisis/models.py
from django.db import models

class AnalisisSentimiento(models.Model):
    texto = models.TextField()
    resultado = models.TextField()
    fecha_analisis = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Análisis de: {self.texto[:50]}"