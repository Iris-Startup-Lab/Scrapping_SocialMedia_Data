# backend/scraping/scripts/twitter_scraper.py
import tweepy
#from backend.scraping.models import DatosWeb  # Vamos a reutilizar este modelo por ahora
from scraping.models import DatosWeb, Tweet  # Vamos a reutilizar este modelo por ahora
from datetime import datetime
import os 
import sys
from dotenv import  load_dotenv
import time 
from datetime import datetime, timedelta

load_dotenv()


# Configuración de la API de Twitter (X)
# Reemplaza con tus propias credenciales de la API de Twitter (X)
twitter_access_token = os.environ.get('twitter_access_token')
twitter_access_token_secret = os.environ.get('twitter_access_token_secret')
twitter_api_key = os.environ.get('twitter_api_key')
twitter_bearer_token = os.environ.get('twitter_bearer_token')
twitter_secret = os.environ.get('twitter_api_secret')


# Autenticación con la API de Twitter (X)
auth = tweepy.OAuthHandler(twitter_api_key, twitter_secret)
auth.set_access_token(twitter_access_token, twitter_access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)
client = tweepy.Client(twitter_bearer_token)


def get_tweet_details(tweet_id, max_results=10):
    try:
        tweets = client.search_recent_tweets(query = tweet_id, 
                                             expansions=["author_id"], 
                                             user_fields=["username"],
                                             max_results=max_results)
        users = {u["id"]: u for u in tweets.includes['users']}
        for tweet in tweets.data:
            tweet_url = f"https://twitter.com/{tweet.author_id}/status/{tweet.id}"
            tweet_id = tweet.id,
            username = users[tweet.author_id]["username"] if tweet.author_id in users else None,
            tweet_text = tweet.text,
            timestamp = tweet.created_at,
            timestamp_iso = timestamp.isoformat()
            #today = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            # Guardar los datos en la base de datos
            Tweet.objects.create(
                tweet_id = tweet_id,
                user_id = tweet.author_id,
                username = username,
                created_at = timestamp_iso,
                full_text = tweet_text,
                url = tweet_url#,
                #fecha_scraping = today
            )
        print(f"Scraping listo para el tweet: {tweet_id}")
    except tweepy.TweepyException as e:
        print(f"Error al scraping del tweet: {e}") 



def scrape_tweets(query, count=10):
    try:
        tweets = client.search_recent_tweets(q=query, count=count, tweet_mode='extended', 
                                             result_type='popular', lang='es')
        for tweet in tweets:
            tweet_url = f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.author_id}"
            tweet_text = tweet.text
            timestamp = tweet.created_at
            # Guardar los datos en la base de datos
            DatosWeb.objects.create(
                texto=tweet_text,
                url=tweet_url,
                fecha_scraping=timestamp
            )
        print(f"Scraping listo {len(tweets)} tweets para la consulta: '{query}'")
    except tweepy.TweepyException as e:
        print(f"Error al scraping tweets: {e}")

if __name__ == "__main__":
    id_tweet = '1234567890'  # Reemplaza con el ID del tweet que deseas obtener
    #query_term = "python"  
    num_tweets = 10
    get_tweet_details(id_tweet, num_tweets)