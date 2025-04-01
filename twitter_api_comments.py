import tweepy
import time 
import os
from dotenv import load_dotenv
import pandas as pd



def getCommentsFromTweet(tweet_id, bearer_token, max_results=10):
    client = tweepy.Client(bearer_token=bearer_token)
    listTweets = client.search_recent_tweets(
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
            'author_id': tweet.author_id,
            'username': users[tweet.author_id]["username"] if tweet.author_id in users else None
        }
        tweets_list.append(tweet_info)
    df_tweets = pd.DataFrame(tweets_list)
    return df_tweets

        

def getCommentsFromTweet(tweet_id, bearer_token, max_results=50):
    """
    Obtiene los comentarios de un tweet específico.

    Args:
        tweet_id (str): El ID del tweet.
        bearer_token (str): El token de portador de la API de X.
        max_results (int, optional): El número máximo de comentarios a recuperar. Defaults to 50.

    Returns:
        pandas.DataFrame: Un DataFrame con los comentarios del tweet, o None si hay un error.
    """
    client = tweepy.Client(bearer_token=bearer_token)
    tweets_list = []
    next_token = None

    while True:
        try:
            listTweets = client.search_recent_tweets(
                query=f"conversation_id:{tweet_id}",
                expansions=["author_id"],
                user_fields=["username"],
                max_results=max_results,
                next_token=next_token,
            )

            if listTweets.data:
                tweets = listTweets.data
                users = {u["id"]: u for u in listTweets.includes["users"]} if listTweets.includes and listTweets.includes["users"] else {}

                for tweet in tweets:
                    tweet_info = {
                        'tweet_id': tweet.id,
                        'text': tweet.text,
                        'author_id': tweet.author_id,
                        'username': users[tweet.author_id]["username"] if tweet.author_id in users else None,
                    }
                    tweets_list.append(tweet_info)

                if "next_token" in listTweets.meta:
                    next_token = listTweets.meta["next_token"]
                    time.sleep(1) #To avoid rate limiting.
                else:
                    break
            else:
                break

        except tweepy.TooManyRequests:
            print("Rate limit exceeded. Waiting...")
            time.sleep(15) 
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    if tweets_list:
        df_tweets = pd.DataFrame(tweets_list)
        return df_tweets
    else:
        return None
