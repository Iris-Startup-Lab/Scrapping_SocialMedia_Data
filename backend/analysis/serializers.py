# backend/analisis/serializers.py
from rest_framework import serializers
from .models import AnalisisSentimiento
from .models import ResultadoAnalisisSentimiento

class AnalisisSentimientoSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalisisSentimiento
        fields = '__all__'

class ResultadoAnalisisSentimientoSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResultadoAnalisisSentimiento
        fields = '__all__'