# backend/chat/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('obtener-respuesta/', views.obtener_gemini_response),
]