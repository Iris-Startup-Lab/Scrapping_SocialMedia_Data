from django.shortcuts import render

# Create your views here.
# backend/analisis/views.py
from rest_framework import generics
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import AnalisisSentimiento, ResultadoAnalisisSentimiento
from .serializers import AnalisisSentimientoSerializer,  ResultadoAnalisisSentimientoSerializer

class AnalisisSentimientoList(generics.ListAPIView):
    queryset = AnalisisSentimiento.objects.all()
    serializer_class = AnalisisSentimientoSerializer

@api_view(['POST'])
def obtener_resultados_analisis_tweets(request):
    tweet_ids = request.data.get('tweet_ids', [])
    resultados = ResultadoAnalisisSentimiento.objects.filter(tweet_id__in=tweet_ids)
    serializer = ResultadoAnalisisSentimientoSerializer(resultados, many=True)
    return Response(serializer.data)    