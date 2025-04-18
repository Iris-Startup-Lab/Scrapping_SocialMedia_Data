from django.shortcuts import render

# Create your views here.
# backend/chat/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from chat.scripts.gemini_connector import obtener_respuesta_gemini
from .serializers import RespuestaGeminiSerializer
from .models import RespuestaGemini

@api_view(['POST'])

def obtener_gemini_response(request):
    """ En base a una función anterior, genera la respuesta de Gemini 

     Parámetros
     -----------------
     request : str
         Una cadena con cualquier pregunta para el llm

     Returns/Retorna 
     ----------
     Una cadena simple o compleja de la respuesta del llm, 
     al existir error pregunta sobre que se requiere
     
    """
    pregunta = request.data.get('pregunta')
    if pregunta:
        respuesta_texto = obtener_respuesta_gemini(pregunta)
        # Opcional: Serializar y devolver la respuesta guardada en el modelo
        try:
            respuesta_obj = RespuestaGemini.objects.latest('fecha_respuesta')
            serializer = RespuestaGeminiSerializer(respuesta_obj)
            return Response(serializer.data)
        except RespuestaGemini.DoesNotExist:
            return Response({'respuesta': respuesta_texto})
    return Response({'error': 'Se requiere la pregunta.'}, status=400)





