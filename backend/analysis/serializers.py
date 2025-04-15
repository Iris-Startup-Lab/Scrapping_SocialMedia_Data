# backend/analisis/serializers.py
from rest_framework import serializers
from .models import AnalisisSentimiento

class AnalisisSentimientoSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalisisSentimiento
        fields = '__all__'