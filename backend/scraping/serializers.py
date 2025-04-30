# backend/scraping/serializers.py
from rest_framework import serializers
from .models import Tweet, DatosWeb

class DatosWebSerializer(serializers.ModelSerializer):
    class Meta:
        #model = DatosWeb
        model = Tweet
        fields = '__all__' 