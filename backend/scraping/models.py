# backend/scraping/models.py
from django.db import models

class DatosWeb(models.Model):
    texto = models.TextField()
    url = models.URLField()
    fecha_scraping = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.texto[:50]
    
class Tweet(models.Model):
    tweet_id = models.CharField(max_length=255, unique=True)
    user_id = models.CharField(max_length=255)
    username = models.CharField(max_length=255)
    created_at = models.DateTimeField(null=True, blank=True)
    full_text = models.TextField()
    url = models.URLField()
    fecha_scraping = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"{self.username} - {self.tweet_id}"
    

class TweetsList(models.Model):
    tweet_id = models.CharField(max_length=255, unique=True)
    user_id = models.CharField(max_length=255)
    username = models.CharField(max_length=255)
    created_at = models.DateTimeField()
    full_text = models.TextField()
    url = models.URLField()
    fecha_scraping = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"{self.username} - {self.tweet_id}"
