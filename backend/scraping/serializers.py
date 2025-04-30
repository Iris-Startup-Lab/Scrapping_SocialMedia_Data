# backend/scraping/serializers.py
from rest_framework import serializers
from .models import DatosWeb

class DatosWebSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatosWeb
        fields = '__all__' 