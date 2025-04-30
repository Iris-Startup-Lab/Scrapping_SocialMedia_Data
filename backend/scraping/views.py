from django.shortcuts import render
# Create your views here.
# backend/scraping/views.py
from rest_framework import generics
from .models import DatosWeb, Tweet
from .serializers import DatosWebSerializer

class DatosWebList(generics.ListAPIView):
    #queryset = DatosWeb.objects.all()
    queryset = Tweet.objects.all()
    serializer_class = DatosWebSerializer