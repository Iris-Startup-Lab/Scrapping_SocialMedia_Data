import tweepy
import os 
from dotenv import load_dotenv
import time 
import pandas as pd
from pathlib import Path

load_dotenv()

#### Main Function to get Tweets 

def getCommentsFromTweet(tweet_id, bearer_token, max_results=10):
    client = tweepy.Client(bearer_token=bearer_token)   
    listTweets = client.search_recent_tweets(
        #### Getting the data from the API
        query=f"conversation_id:{tweet_id}",
        expansions=["author_id"],  
        user_fields=["username"],  
        max_results=max_results
    )
    tweets = listTweets.data
    users = {u["id"]: u for u in listTweets.includes["users"]}
    tweets_list = []
    for tweet in tweets:
        tweet_info = {
            'tweet_id': tweet.id,
            'text': tweet.text,
            'tweet_created_at': tweet.created_at, 
            'author_id': tweet.author_id,
            'tweet_id': tweet.id,
            'author_id': users[tweet.author_id]['username'] if tweet.author_id in users else None                                                
        }
        tweets_list.append(tweet_info)
    df_tweets = pd.DataFrame(tweets_list)
    return df_tweets
