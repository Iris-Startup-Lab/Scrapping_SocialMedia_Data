# data_fetchers.py
import logging
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import tweepy
from googleapiclient.discovery import build
import googlemaps
import praw
from google_play_scraper import reviews_all, Sort

from config import (
    YOUTUBE_API_KEY, MAPS_API_KEY, TWITTER_BEARER_TOKEN,
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
)

logger = logging.getLogger(__name__)

def get_tweets(query: str) -> pd.DataFrame:
    if not TWITTER_BEARER_TOKEN:
        return pd.DataFrame({'Error': ["Bearer Token de Twitter no configurado."]})
    try:
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        # Lógica simplificada para buscar tweets recientes
        response = client.search_recent_tweets(query=query, tweet_fields=['created_at', 'public_metrics', 'lang'], max_results=100)
        if not response.data:
            return pd.DataFrame({'Mensaje': ["No se encontraron tweets."]})
        tweets_list = [{'text': tweet.text, 'created_at': tweet.created_at, **tweet.public_metrics, 'lang': tweet.lang} for tweet in response.data]
        df = pd.DataFrame(tweets_list)
        df['origin'] = 'twitter'
        return df
    except Exception as e:
        logger.error(f"Error en get_tweets: {e}")
        return pd.DataFrame({'Error': [f"Error al obtener tweets: {e}"]})

def get_youtube_comments(video_url: str) -> pd.DataFrame:
    if not YOUTUBE_API_KEY:
        return pd.DataFrame({'Error': ["Clave API de YouTube no configurada."]})
    video_id_match = re.search(r"v=([a-zA-Z0-9_-]+)", video_url)
    if not video_id_match:
        return pd.DataFrame({'Error': ["URL de YouTube no válida."]})
    video_id = video_id_match.group(1)
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100)
        response = request.execute()
        comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in response['items']]
        df = pd.DataFrame(comments, columns=['comment'])
        df['origin'] = 'youtube'
        return df
    except Exception as e:
        logger.error(f"Error en get_youtube_comments: {e}")
        return pd.DataFrame({'Error': [f"Error al obtener comentarios de YouTube: {e}"]})

def get_maps_reviews(query: str):
    if not MAPS_API_KEY:
        return pd.DataFrame({'Error': ["Clave API de Google Maps no configurada."]}), None
    gmaps = googlemaps.Client(key=MAPS_API_KEY)
    try:
        places_result = gmaps.places(query=query)
        if not places_result or not places_result.get('results'):
            return pd.DataFrame({'Mensaje': ["No se encontraron lugares."]}), None
        place_id = places_result['results'][0]['place_id']
        place_details = gmaps.place(place_id=place_id, fields=['review', 'geometry'])
        reviews = place_details.get('result', {}).get('reviews', [])
        if not reviews:
            return pd.DataFrame({'Mensaje': ["El lugar no tiene reseñas."]}), None
        df = pd.DataFrame(reviews)
        df.rename(columns={'text': 'comment'}, inplace=True)
        df['origin'] = 'maps'
        coords = place_details.get('result', {}).get('geometry', {}).get('location')
        return df, coords
    except Exception as e:
        logger.error(f"Error en get_maps_reviews: {e}")
        return pd.DataFrame({'Error': [f"Error al obtener reseñas de Google Maps: {e}"]}), None

def get_reddit_comments(url: str) -> pd.DataFrame:
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        return pd.DataFrame({'Error': ["Credenciales de Reddit API no configuradas."]})
    try:
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)
        submission = reddit.submission(url=url)
        submission.comments.replace_more(limit=0)
        comments = [comment.body for comment in submission.comments.list()]
        df = pd.DataFrame(comments, columns=['comment'])
        df['origin'] = 'reddit'
        return df
    except Exception as e:
        logger.error(f"Error en get_reddit_comments: {e}")
        return pd.DataFrame({'Error': [f"Error al obtener comentarios de Reddit: {e}"]})

def get_playstore_reviews(app_url: str) -> pd.DataFrame:
    app_id_match = re.search(r'id=([a-zA-Z0-9._]+)', app_url)
    if not app_id_match:
        return pd.DataFrame({'Error': ["URL de Play Store no válida."]})
    app_id = app_id_match.group(1)
    try:
        all_reviews = reviews_all(app_id, lang='es', country='mx', sort=Sort.NEWEST)
        if not all_reviews:
            return pd.DataFrame({'Mensaje': ["No se encontraron reseñas."]})
        df = pd.DataFrame(all_reviews)
        df.rename(columns={'content': 'comment'}, inplace=True)
        df['origin'] = 'playstore'
        return df
    except Exception as e:
        logger.error(f"Error en get_playstore_reviews: {e}")
        return pd.DataFrame({'Error': [f"Error al obtener reseñas de Play Store: {e}"]})

def get_wikipedia_text(url: str) -> pd.DataFrame:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p') if p.get_text().strip()]
        df = pd.DataFrame(paragraphs, columns=['text'])
        df['origin'] = 'wikipedia'
        return df
    except Exception as e:
        logger.error(f"Error en get_wikipedia_text: {e}")
        return pd.DataFrame({'Error': [f"Error al obtener texto de Wikipedia: {e}"]})

def get_generic_webpage_text(url: str) -> pd.DataFrame:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p') if p.get_text().strip()]
        df = pd.DataFrame(paragraphs, columns=['text'])
        df['origin'] = 'generic_webpage'
        return df
    except Exception as e:
        logger.error(f"Error en get_generic_webpage_text: {e}")
        return pd.DataFrame({'Error': [f"Error al acceder a la página web: {e}"]})