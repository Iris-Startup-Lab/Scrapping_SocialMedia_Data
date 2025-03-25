import tweepy
import os
from dotenv import load_dotenv
import pandas as pd



def getCommentsFromTweet(tweet_id, bearer_token):
    client = tweepy.Client(bearer_token=bearer_token)
    listTweets = client.search_recent_tweets(
        query=f"conversation_id:{tweet_id}",
        expansions=["author_id"],  # Información del autor del tweet o xeet?
        user_fields=["username"],  # Obtener el @ del autor
        max_results=50
    )
    tweets = listTweets.data
    users = {u["id"]: u for u in listTweets.includes["users"]}
    tweets_list = []
    for tweet in tweets:
        tweet_info = {
            'tweet_id': tweet.id,
            'text': tweet.text,
            'author_id': tweet.author_id,
            'username': users[tweet.author_id]["username"] if tweet.author_id in users else None
        }
        tweets_list.append(tweet_info)
    df_tweets = pd.DataFrame(tweets_list)
    return df_tweets

        

    
