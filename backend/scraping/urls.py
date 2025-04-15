# backend/scraping/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('datos-web/', views.DatosWebList.as_view()),
]