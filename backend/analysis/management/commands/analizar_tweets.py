# backend/analisis/management/commands/analizar_tweets.py
from django.core.management.base import BaseCommand
from scraping.models import Tweet
from analysis.scripts.sentiment_analyzer import analizar_sentimiento_hibrido

class Command(BaseCommand):
    help = 'Analiza el sentimiento de los tweets almacenados'
    def handle(self, *args, **options):
        tweets_sin_analizar = Tweet.objects.filter(analisis_sentimientos__isnull=True)
        self.stdout.write(f'Se encontraron {tweets_sin_analizar.count()} tweets sin analizar.')
        for tweet in tweets_sin_analizar:
            self.stdout.write(f'Analizando el tweet: {tweet.tweet_id} - "{tweet.full_text[:50]}..."')
            analizar_sentimiento_hibrido(tweet)
        self.stdout.write(self.style.SUCCESS('Análisis de sentimientos completado.'))
