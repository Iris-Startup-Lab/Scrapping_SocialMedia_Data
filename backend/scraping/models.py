# backend/scraping/models.py
from django.db import models

class DatosWeb(models.Model):
    texto = models.TextField()
    url = models.URLField()
    fecha_scraping = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.texto[:50]