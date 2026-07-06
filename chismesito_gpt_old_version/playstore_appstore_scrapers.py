# -*- coding: utf-8 -*-
## Iris Startup Lab 
'''
<(*)
  ( >)
  /|
'''

#-------------------------------------------------------------
######### Social Media Downloader Shiny App ######
######### VERSION 0.5 ######
######### Authors Fernando Dorantes Nieto
#-------------------------------------------------------------


#"cd .\Local\scripts\Social_media_comments\shiny_app\iris_social_media_downloader"
#### Add Pinecone and the button of comments
############# PLAYSTORE SCRAPERS ###############
import re 
from google_play_scraper import app as play_app, reviews as play_reviews, Sort, reviews_all, search
