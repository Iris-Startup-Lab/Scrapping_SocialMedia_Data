# backend/scraping/scripts/twitter_scraper.py
import tweepy
import os
from dotenv import load_dotenv
load_dotenv()

def authenticate_twitter_api():
    consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
    consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
    access_token = os.getenv("TWITTER_ACCESS_TOKEN")
    access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

def get_tweet_details(api, tweet_id):
    try:
        tweet = api.get_status(tweet_id, tweet_mode='extended')
        return {
            'id_str': tweet.id_str,
            'created_at': tweet.created_at.isoformat(),
            'full_text': tweet.full_text,
            'user': {
                'id_str': tweet.user.id_str,
                'screen_name': tweet.user.screen_name
            },
            'retweet_count': tweet.retweet_count,
            'favorite_count': tweet.favorite_count,
            'entities': tweet.entities,
            'extended_entities': getattr(tweet, 'extended_entities', None)
        }
    except tweepy.TweepyException as e:
        print(f"Error al obtener detalles del tweet {tweet_id}: {e}")
        return None

if __name__ == '__main__':
    api = authenticate_twitter_api()
    if api:
        tweet_id_to_get = "1460328651708331008"  # Ejemplo de tweet ID
        tweet_info = get_tweet_details(api, tweet_id_to_get)
        if tweet_info:
            print(tweet_info['full_text'])
            print(f"URL: https://x.com/{tweet_info['user']['screen_name']}/status/{tweet_info['id_str']}")