# backend/analisis/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('resultados-analisis/', views.AnalisisSentimientoList.as_view()),
    path('resultados-analisis-tweets/', views.obtener_resultados_analisis_tweets, name='obtener_resultados_analisis_tweets')
]