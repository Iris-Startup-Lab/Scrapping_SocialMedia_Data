# backend/analisis/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('resultados-analisis/', views.AnalisisSentimientoList.as_view()),
]