import pandas as pd
import numpy as np
import re 
from GetCredentials import *
from twitter_api_comments import getCommentsFromTweet

from youtube_api_comments import commentsFromVideo, commentstoDf


def getSocialMediaInfo(url):
    if re.search('youtube', url):
        return getCommentsFromTweet(url, bearer_token=twitter_bearer_token)
    elif re.search('x.com', url):
        return commentstoDf(video_id=url, api_key=google_api_key)


