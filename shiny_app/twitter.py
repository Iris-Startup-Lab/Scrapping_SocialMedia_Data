import tweepy
import os 
from dotenv import load_dotenv
import time 
load_dotenv()

#### Main Function to get Tweets 

def getCommentsFromTweet(tweet_id, bearer_token, max_results=10):
    client = tweepy.Client(bearer_token=bearer_token)   