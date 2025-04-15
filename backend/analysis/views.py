from django.shortcuts import render

# Create your views here.
# backend/analisis/views.py
from rest_framework import generics
from .models import AnalisisSentimiento
from .serializers import AnalisisSentimientoSerializer

class AnalisisSentimientoList(generics.ListAPIView):
    queryset = AnalisisSentimiento.objects.all()
    serializer_class = AnalisisSentimientoSerializer