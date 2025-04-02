import os 
from googleapiclient.discovery import build
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

#### Starting the env

def commentsFromVideo(video_id, api_key):

    #google_api_key = os.getenv("google_api_key")
    google_api_key = api_key
    youtube = build("youtube", "v3", 
                    developerKey=google_api_key)
    comments = []
    next_page_token = None
    while True:
        response = youtube.commentThreads().list(
            part = 'snippet',
            videoId = video_id,
            maxResults = 100,
            pageToken = next_page_token
        ).execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(
                {
                'author': comment['authorDisplayName'],
                'text': comment['textDisplay'],
                'likes': comment['likeCount'],
                'published_at': comment['publishedAt']
             }
        )
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    return comments
    



def commentstoDf(video_id, api_key):
    comments = commentsFromVideo(video_id, api_key)
    df = pd.DataFrame(comments)
    return df

### ending youtube api commments
