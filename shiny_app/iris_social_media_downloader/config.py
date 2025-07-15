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
####### This script hosts the main objects from the environment
####### at the same time the code provides path directions and web browser paths for chrome extension
###### This script must be run in all the dependencies that needs some secrets or something related
#-------------------------------------------------------------

# config.py
import os
from pathlib import Path
from dotenv import load_dotenv #### Load the environment


load_dotenv()
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
MAPS_API_KEY = os.environ.get("MAPS_API_KEY")
### X formerly Twitter Zone
TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY")
TWITTER_API_SECRET = os.environ.get("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.environ.get("TWITTER_ACCESS_SECRET")
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT")
DETECT_LANGUAGE_API_KEY = os.environ.get("DETECT_LANGUAGE_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "iris-gemini-chat") 
EMBEDDING_DIMENSION = 768 # Para models/embedding-001 de Gemini
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_KEY_PSQL = os.getenv("SUPABASE_KEY_PSQL")
SUPABASE_URL_PSQL = os.getenv("SUPABASE_URL_PSQL")
SUPABASE_USER_PSQL = os.getenv("SUPABASE_USER")

OXYLABS_SCRAPER_API_ENDPOINT = os.getenv("OXYLABS_ENDPOINT")
OXYLABS_API_USER = os.getenv("OXYLABS_USER")
OXYLABS_API_PASS = os.getenv("OXYLABS_PASS")

PG_HOST = SUPABASE_URL_PSQL
PG_PORT = "5432"
PG_DBNAME =  "postgres"
PG_USER = SUPABASE_USER_PSQL
PG_PASSWORD = SUPABASE_KEY_PSQL


# Rutas de ChromeDriver y Chrome Binary
if os.name == 'nt':  # Windows
    CHROME_DRIVER_PATH = os.getenv("CHROME_DRIVER_PATH_WINDOWS")
    CHROME_BINARY_PATH = os.getenv("CHROME_BINARY_PATH_WINDOWS")
    PLATFORM_STEALTH = "Win32"
elif os.name == 'posix':  # Linux o macOS
    CHROME_DRIVER_PATH = os.getenv("CHROME_DRIVER_PATH_LINUX")
    CHROME_BINARY_PATH = os.getenv("CHROME_BINARY_PATH_LINUX")
    PLATFORM_STEALTH = "Linux" 
else:  # Fallback
    CHROME_DRIVER_PATH = '/usr/bin/chromedriver'
    CHROME_BINARY_PATH = '/usr/bin/google-chrome'
    PLATFORM_STEALTH = "Linux"

# Lista de User-Agents para el scraper
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
]

times = [5,7,4,3] ## Some random numbers 
# Directorio del script actual
here = Path(__file__).parent

