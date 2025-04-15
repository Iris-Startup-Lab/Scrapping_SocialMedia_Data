# backend/scraping/serializers.py
from rest_framework import serializers
from .models import DatosWeb

class DatosWebSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatosWeb
        fields = '__all__'  # Incluye todos los campos del modelo
        # Si solo quieres ciertos campos, especifícalos en una lista:
        # fields = ['id', 'texto', 'url', 'fecha_scraping']