# backend/chat/models.py
from django.db import models

class RespuestaGemini(models.Model):
    pregunta = models.TextField()
    respuesta = models.TextField()
    fecha_respuesta = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Pregunta: {self.pregunta[:50]}"