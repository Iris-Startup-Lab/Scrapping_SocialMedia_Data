import os 
from dotenv import load_dotenv


load_dotenv()


google_api_key = os.getenv('google_api_key')
twitter_api_key = os.getenv('twitter_api_key')
twitter_api_secret = os.getenv('twitter_api_secret')
twitter_access_token = os.getenv('twitter_access_token')
twitter_access_token_secret = os.getenv('twitter_access_token_secret')
twitter_bearer_token = os.getenv('twitter_bearer_token')
