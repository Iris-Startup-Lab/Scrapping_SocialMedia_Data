# backend/chat/serializers.py
from rest_framework import serializers
from .models import RespuestaGemini

class RespuestaGeminiSerializer(serializers.ModelSerializer):
    class Meta:
        model = RespuestaGemini
        fields = '__all__'