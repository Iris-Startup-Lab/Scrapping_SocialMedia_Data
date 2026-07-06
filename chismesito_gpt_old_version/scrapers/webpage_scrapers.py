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
###### This script is hosting the codes of the scrapers
#-------------------------------------------------------------

import os
import pandas as pd
import googlemaps
import re
import tweepy
import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
from googleapiclient.discovery import build ## Youtube

from sqlalchemy import create_engine
import time 

load_dotenv()
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
MAPS_API_KEY = os.environ.get("MAPS_API_KEY")
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


def _maps_comments(maps_query: str) -> pd.DataFrame:
        """
        Adapta la lógica de searchMapsCommentsQuery del prompt original.
        Busca lugares cercanos por palabra clave y obtiene sus comentarios.
        """
        if not MAPS_API_KEY: return pd.DataFrame({'Error': ["Clave API de Google Maps no configurada."]})
        
        gmaps = googlemaps.Client(MAPS_API_KEY)
        comments_list = []

        try:
            # 1. Buscar lugares cercanos por la palabra clave
            # geocoder.ip('me') no está disponible globalmente en app.py,
            # se asume una ubicación por defecto o se podría integrar un servicio de geocodificación.
            # Para simplificar, usaremos una ubicación central de CDMX o se podría hacer configurable.
            # O se podría usar el input.maps_query() si es una dirección.
            # Si maps_query es una URL con place_id, se prioriza.
            place_id = None
            if maps_query.startswith("http") and "place_id" in maps_query:
                match = re.search(r'place_id=([^&]+)', maps_query)
                if match: place_id = match.group(1)
            
            if place_id:
                # Si se proporciona un place_id, obtener directamente los detalles de ese lugar
                places_data = gmaps.place(place_id=place_id,
                                        fields=['name', 'rating', 'review', 'formatted_address', 'geometry', 'place_id'],
                                        language='es')
                if places_data.get('result'):
                    result_data = places_data['result']
                    placeName = result_data.get('name', 'N/A')
                    lat = result_data['geometry']['location']['lat'] if result_data.get('geometry') and result_data['geometry'].get('location') else None
                    long = result_data['geometry']['location']['lng'] if result_data.get('geometry') and result_data['geometry'].get('location') else None
                    reviews_data = result_data.get('reviews', [])
                    
                    current_place_comments = []
                    for review in reviews_data:
                        current_place_comments.append({
                            'author': review.get('author_name', 'N/A'), 
                            'comment': review.get('text', ''), 
                            'rating': review.get('rating', 'N/A'),
                            'lat': lat,
                            'long': long,
                            'place_name': placeName,
                            'place_id': place_id,
                            'query': maps_query,
                            'source': 'Google Maps'
                        })
                    if current_place_comments:
                        comments_list.append(pd.DataFrame(current_place_comments))
            else:
                # Si no hay place_id, realizar una búsqueda de lugares cercanos
                # Usar una ubicación por defecto (ej. Ciudad de México) si geocoder.ip('me') no es viable
                # O se podría hacer que el usuario ingrese una ubicación de referencia
                default_location = (19.4326, -99.1332) # Lat/Long de CDMX
                searchPlaces = gmaps.places_nearby(keyword=maps_query, 
                                                location=default_location,
                                                radius=5000) # Radio de 5km
                searchPlacesResults = searchPlaces['results']

                for place in searchPlacesResults:
                    placeName = place.get('name', None)
                    # Filtrar por nombre para asegurar relevancia, como en el prompt original
                    if placeName and re.search(re.escape(maps_query.lower()), placeName.lower(), re.IGNORECASE):
                        placesData = gmaps.place(place_id=place.get('place_id'),
                                                fields=['name', 'rating', 'review', 'formatted_address', 'geometry', 'place_id'],
                                                language='es')
                        current_place_comments = []
                        if placesData.get('result'):
                            result_data = placesData['result']
                            lat = result_data['geometry']['location']['lat'] if result_data.get('geometry') and result_data['geometry'].get('location') else None
                            long = result_data['geometry']['location']['lng'] if result_data.get('geometry') and result_data['geometry'].get('location') else None
                            reviews_data = result_data.get('reviews', [])
                            if reviews_data: 
                                for review in reviews_data:
                                    current_place_comments.append({
                                        'author': review.get('author_name', 'N/A'), 
                                        'comment': review.get('text', ''), 
                                        'rating': review.get('rating', 'N/A'),
                                        'lat': lat,
                                        'long': long,
                                        'place_name': placeName,
                                        'place_id': place.get('place_id', None),
                                        'query': maps_query,
                                        'source': 'Google Maps'
                                    })
                        if current_place_comments:
                            comments_list.append(pd.DataFrame(current_place_comments))

        except Exception as e:
            logger.error(f"Error al obtener comentarios de Google Maps para '{maps_query}': {e}", exc_info=True)
            return pd.DataFrame({"Error": [f"Error al obtener comentarios de Google Maps: {e}"]})

        if comments_list:
            final_df = pd.concat(comments_list, ignore_index=True)
            final_df['origin'] = 'maps' # Asegurar la columna 'origin'
            return final_df
        return pd.DataFrame({'Mensaje': [f"No se encontraron reviews para '{maps_query}'."]})



def _get_youtube_channels_and_comments(query: str, max_results_comments: int = 10) -> pd.DataFrame:
    """
    Adapta la lógica de searchYoutubeQuery del prompt original.
    Busca canales de YouTube por query, obtiene videos del canal más relevante
    y luego los comentarios de esos videos.
    """
    if not YOUTUBE_API_KEY:
        return pd.DataFrame({'Error': ["Clave API de YouTube no configurada."]})
    
    youtube_service = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    comments_list = []

    try:
        # 1. Buscar canales de YouTube por la query
        request_channels = youtube_service.search().list(
            q=query,
            type='channel',
            part='snippet',
            maxResults=10 # Buscar hasta 5 canales
        )
        response_channels = request_channels.execute()
        
        canales_similares = []
        for item in response_channels.get('items', []):
            channel_title_search = item['snippet']['title'].lower()
            if re.search(re.escape(query.lower()), channel_title_search, re.IGNORECASE):
                canales_similares.append({
                    'channelId': item['id']['channelId'],
                    'title': item['snippet']['title']
                })
        
        if not canales_similares:
            logger.warning(f"No se encontraron canales similares a '{query}'.")
            return pd.DataFrame(columns=['author', 'comment', 'published_at', 'video_id', 'video_title', 'query', 'source', 'channelId', 'official_channel', 'channel_title', 'origin'])

        # Tomar el primer canal similar encontrado
        selected_channel = canales_similares[0]
        channel_id = selected_channel['channelId']
        channel_title = selected_channel['title']

        # Determinar si el canal es oficial (ej. verificado o vinculado)
        testStatus = youtube_service.channels().list(part='status', id=channel_id).execute()
        is_official = testStatus.get('items', [{}])[0].get('status', {}).get('isLinked', False)

        # 2. Buscar videos del canal seleccionado
        request_videos = youtube_service.search().list(
            channelId=channel_id,
            part='snippet',
            maxResults=20, # Obtener hasta 20 videos
            order='date'
        )
        response_videos = request_videos.execute()
        
        for video in response_videos.get('items', []):
            video_id = video['id'].get('videoId')
            video_title = video['snippet']['title']
            if video_id:
                comments = _get_comments_from_video_helper(video_id, youtube_service, max_results_comments)
                if comments:
                    df_comments = pd.DataFrame(comments)
                    df_comments['video_id'] = video_id
                    df_comments['video_title'] = video_title
                    df_comments['query'] = query
                    df_comments['source'] = 'YouTube'
                    df_comments['channelId'] = channel_id
                    df_comments['official_channel'] = is_official
                    df_comments['channel_title'] = channel_title
                    comments_list.append(df_comments)

    except Exception as e:
        logger.error(f"Error en _get_youtube_channels_and_comments para '{query}': {e}", exc_info=True)
        return pd.DataFrame({"Error": [f"Error al obtener datos de YouTube: {e}"]})

    if comments_list:
        final_df = pd.concat(comments_list, ignore_index=True)
        final_df['origin'] = 'youtube' # Asegurar la columna 'origin'
        return final_df
    return pd.DataFrame(columns=['author', 'comment', 'published_at', 'video_id', 'video_title', 'query', 'source', 'channelId', 'official_channel', 'channel_title', 'origin'])

def _get_comments_from_video_helper(videoId: str, youtube_service, max_results: int = 20) :
    """
    Función auxiliar para obtener comentarios de un video, usada por _get_youtube_channels_and_comments.
    """
    comments = []
    request = youtube_service.commentThreads().list(
        part='snippet',
        videoId=videoId,
        textFormat='plainText',
        maxResults=max_results
    )
    
    while request:
        response = request.execute()
        for item in response.get('items', []):
            comment_info = {
                'author': item['snippet']['topLevelComment']['snippet']['authorDisplayName'],
                'comment': item['snippet']['topLevelComment']['snippet']['textDisplay'],
                'published_at': item['snippet']['topLevelComment']['snippet']['publishedAt']
            }
            comments.append(comment_info)
        
        request = youtube_service.commentThreads().list_next(request, response)
    
    return comments


def _get_youtube_comments(video_url_or_id: str) -> pd.DataFrame:
    """
    Obtiene comentarios de un video de YouTube específico por su URL o ID.
    """
    if not YOUTUBE_API_KEY:
        return pd.DataFrame({'Error': ["Clave API de YouTube no configurada."]})
    
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    video_id = None
    
    # Extrae el ID del video de varios formatos de URL
    if "v=" in video_url_or_id: 
        video_id = video_url_or_id.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in video_url_or_id: 
        video_id = video_url_or_id.split("youtu.be/")[-1].split("?")[0]
    elif len(video_url_or_id) == 11 and video_url_or_id.isalnum(): # Asume que es un ID si tiene 11 caracteres alfanuméricos
        video_id = video_url_or_id
    
    if not video_id: 
        return pd.DataFrame({'Error': ["URL/ID de YouTube no válido."]})

    try:
        comments = []
        # Realiza la llamada a la API para obtener los hilos de comentarios
        response = youtube.commentThreads().list(
            part='snippet',  
            videoId=video_id, 
            textFormat='plainText', 
            maxResults=10 # Obtiene los 10 comentarios principales
        ).execute()
        
        # Procesa la respuesta
        for item in response.get('items', []):
            comment_snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'author': comment_snippet['authorDisplayName'], 
                'comment': comment_snippet['textDisplay'],
                'video_id': video_id, 
                'published_at': comment_snippet['publishedAt']
            })
        
        if not comments: 
            return pd.DataFrame({'Mensaje': [f"No se encontraron comentarios para el video de YouTube {video_id}."]})

        # Convierte la lista de comentarios a un DataFrame de pandas
        df = pd.DataFrame(comments)
        df['origin'] = "youtube" # Añade la columna de origen para análisis posteriores
        return df
        
    except Exception as e:
        logger.error(f"Error al obtener comentarios de YouTube para '{video_url_or_id}': {e}", exc_info=True)
        return pd.DataFrame({"Error": [f"Error al obtener comentarios de YouTube: {e}"]})


def _get_tweets_from_twitter_api(query_or_url: str) -> pd.DataFrame:
    """
    Obtiene tweets de la API de Twitter (X) basándose en una consulta.
    Soporta búsquedas por hashtag, usuario, URL de tweet o ID de tweet.
    
    Args:
        query_or_url (str): La consulta de búsqueda (ej. "#hashtag", "@usuario",
                            URL de tweet como "https://x.com/user/status/ID", o ID numérico de tweet).

    Returns:
        pd.DataFrame: Un DataFrame de pandas con los tweets encontrados y sus metadatos,
                      incluyendo columnas 'query' y 'source' para el análisis comparativo.
                      Devuelve un DataFrame con un mensaje de error o vacío si no se encuentran datos.
    """
    if not TWITTER_BEARER_TOKEN:
        return pd.DataFrame({"Error": ["Bearer Token de Twitter no configurado."]})

    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
    tweets_list = []
    response_data = None

    tweet_fields = ['created_at', 'public_metrics', 'author_id', 'conversation_id', 'in_reply_to_user_id', 'lang']
    expansions = ['author_id', 'in_reply_to_user_id']
    user_fields = ['username', 'name', 'profile_image_url', 'verified']
    max_results_count = 10
    user_profile_match = re.search(r'(?:https?://(?:www\.)?(?:twitter|x)\.com/)([a-zA-Z0-9_]+)(?:/?(?:\?.*)?(?:#.*)?)?$', query_or_url)
    if user_profile_match:
        username_to_search = user_profile_match.group(1)
        logger.info(f'Detectada URL de perfil. Buscando tweets del usuario: @{username_to_search}')
        user_lookup = client.get_user(username=username_to_search, user_fields=['id'])
        if user_lookup.data:
                user_id = user_lookup.data.id
                response_data = client.get_users_tweets(id=user_id, tweet_fields=tweet_fields, expansions=expansions, user_fields=user_fields, max_results=10)
        else: return pd.DataFrame({"Error": [f'Usuario de Twitter no encontrado a partir de la URL: {query_or_url}']})
    try:
        # Verifica si es una URL o ID de tweet
        if ("x.com/" in query_or_url or "twitter.com/" in query_or_url) and "/status/" in query_or_url:
            match = re.search(r'.*/status/(\d+)', query_or_url)
            if match:
                tweet_id = match.group(1)
                logger.info(f'Buscando tweet por ID: {tweet_id}')
                response_data = client.get_tweet(id=tweet_id, tweet_fields=tweet_fields, expansions=expansions, user_fields=user_fields)
                if response_data.data: response_data.data = [response_data.data] # Envuelve el tweet individual en una lista para consistencia
                else: return pd.DataFrame({"Mensaje": [f"No se encontró el tweet con ID: {tweet_id}"]})
            else:
                return pd.DataFrame({"Error": ["No se pudo extraer el ID del tweet de la URL."]})
        elif query_or_url.startswith("#"):
            logger.info(f'Buscando tweets por hashtag: {query_or_url}')
            response_data = client.search_recent_tweets(query=query_or_url, tweet_fields=tweet_fields, expansions=expansions, user_fields=user_fields, max_results=max_results_count)
        elif query_or_url.startswith("@"):
            username_to_search = query_or_url.lstrip('@')
            logger.info(f'Buscando tweets del usuario: {username_to_search}')
            user_lookup = client.get_user(username=username_to_search, user_fields=['id'])
            if user_lookup.data:
                user_id = user_lookup.data.id
                response_data = client.get_users_tweets(id=user_id, tweet_fields=tweet_fields, expansions=expansions, user_fields=user_fields, max_results=max_results_count)
            else: return pd.DataFrame({"Error": [f'Usuario de Twitter no encontrado: {username_to_search}']})
        elif query_or_url.isdigit(): # Si es solo un ID de tweet numérico
            tweet_id = query_or_url
            logger.info(f'Buscando tweet por ID numérico: {tweet_id}')
            response_data = client.get_tweet(id=tweet_id, tweet_fields=tweet_fields, expansions=expansions, user_fields=user_fields)
            if response_data.data: response_data.data = [response_data.data] # Consistencia
            else: return pd.DataFrame({"Mensaje": [f"No se encontró el tweet con ID: {tweet_id}"]})
        else:
            logger.warning(f"Consulta de Twitter no válida: '{query_or_url}'.")
            return pd.DataFrame({"Error": ["Consulta de Twitter no válida (formato @usuario, #hashtag, ID o URL de tweet)"]})

        if response_data and response_data.data:
            users_data = {u["id"]: u for u in response_data.includes.get('users', [])} if response_data.includes else {}
            for tweet_obj in response_data.data:
                author_info = users_data.get(tweet_obj.author_id, {})
                metrics = tweet_obj.public_metrics if tweet_obj.public_metrics else {}
                tweets_list.append({
                    'tweet_id': tweet_obj.id, 'text': tweet_obj.text, 'author_id': tweet_obj.author_id,
                    'username': author_info.get("username", "N/A"), 'author_name': author_info.get("name", "N/A"),
                    'author_verified': author_info.get("verified", False), 'created_at': tweet_obj.created_at,
                    'like_count': metrics.get('like_count', 0), 'retweet_count': metrics.get('retweet_count', 0),
                    'reply_count': metrics.get('reply_count', 0), 'quote_count': metrics.get('quote_count', 0),
                    'impression_count': metrics.get('impression_count', 0), 'conversation_id': tweet_obj.conversation_id,
                    'in_reply_to_user_id': tweet_obj.in_reply_to_user_id, 'lang': tweet_obj.lang
                    ,'query': query_or_url, # Añadir la query original para el contexto de comparación
                    'source': 'Twitter' # Añadir la fuente para el análisis comparativo
                })
        
        if not tweets_list: return pd.DataFrame({'Mensaje': [f"No se encontraron tweets para la consulta '{query_or_url}' o la respuesta no contenía datos."]})
        df = pd.DataFrame(tweets_list)
        df['origin'] = "twitter" # Columna 'origin' para identificar la plataforma en análisis posteriores
        return df
        
    except tweepy.TweepyException as e:
        error_message = str(e)
        if hasattr(e, 'api_errors') and e.api_errors and isinstance(e.api_errors, list) and e.api_errors[0]:
            api_error = e.api_errors[0]
            if isinstance(api_error, dict): 
                error_message = api_error.get('detail', error_message)
                if 'title' in api_error: error_message = f"{api_error['title']}: {error_message}"
            elif hasattr(api_error, 'message'): error_message = api_error.message
        elif hasattr(e, 'response') and e.response is not None:
             try: error_details = e.response.json(); error_message = error_details.get('detail', error_details.get('title', str(e)))
             except ValueError: error_message = e.response.text if e.response.text else str(e)
        logger.error(f"TweepyException para '{query_or_url}': {error_message}", exc_info=True)
        return pd.DataFrame({"Error": [f"Error de API de Twitter: {error_message}"]})
    except Exception as e:
        logger.error(f"Error general en _get_tweets_from_twitter_api para '{query_or_url}': {e}", exc_info=True)
        return pd.DataFrame({"Error": [f"Error general al obtener tweets: {e}"]})


# --- Twitter API Account Rotation Setup ---
# Estas variables y funciones se inicializan una vez al inicio de la aplicación
# y gestionan la rotación de cuentas de Twitter (X) para evitar límites de tasa.

def getXAccounts():
    """
    Obtiene las credenciales de las cuentas de Twitter (X) desde la base de datos.
    Asume que las variables de entorno para la conexión a Supabase están configuradas.
    """
    USER = 'postgres.rdqtsoydvgxdbbvhnmlk'
    PASSWORD = os.getenv("SUPABASE_KEY_PSQL")
    HOST = os.getenv("SUPABASE_URL_PSQL")
    PORT = 5432
    DBNAME = 'postgres'

    query_sql = """
    select
        xa.*
    from
        iris_scraper.x_accounts xa
    """

    try:
        engine = create_engine(f'postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}/{DBNAME}')
        dfFb = pd.read_sql(query_sql, engine)
        return dfFb
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos o al obtener cuentas de X: {e}")
        return pd.DataFrame() # Retorna un DataFrame vacío en caso de error

#xaccountsdf = getXAccounts()

"""
# Asegura que el DataFrame no esté vacío antes de procesar
if not xaccountsdf.empty:
    # Asegura que las columnas coincidan con el orden esperado de la consulta SQL
    xaccountsdf.columns = ['name', 'bearer_token', 'api_key', 'api_secret', 'access_token', 'access_token_secret', 'created_at', 'id']
    xaccountsdf['calls_made'] = 0 # Inicializa el contador de llamadas para cada cuenta
    twitter_accounts = xaccountsdf.to_dict(orient='records')
else:
    logger.warning("No se pudieron cargar las cuentas de Twitter (X) desde la base de datos. Las funciones de Twitter podrían no operar.")
    twitter_accounts = [] # Lista vacía si no hay cuentas
"""
twitter_accounts = None 
current_account_index = 0
MAX_CALLS_PER_ACCOUNT = 100 # Límite de llamadas por cuenta antes de rotar

def _switch_to_next_twitter_account():
    """Pasa a la siguiente cuenta de API en la lista."""
    global current_account_index
    if not twitter_accounts:
        logger.warning("No hay cuentas de Twitter configuradas para rotar.")
        return
    previous_index = current_account_index
    current_account_index = (current_account_index + 1) % len(twitter_accounts)
    logger.info(f"Cambiando de cuenta de API de Twitter: de '{twitter_accounts[previous_index]['name']}' a '{twitter_accounts[current_account_index]['name']}'")

def _get_current_twitter_clients():
    """
    Obtiene los clientes de Tweepy (v1.1 y v2) para la cuenta activa.
    Maneja la rotación de cuentas si la actual está agotada.
    """
    global current_account_index
    if not twitter_accounts:
        logger.error("No hay cuentas de Twitter configuradas.")
        return None, None

    account = twitter_accounts[current_account_index]

    # Si la cuenta actual ha alcanzado su límite, intenta cambiar a la siguiente
    if account['calls_made'] >= MAX_CALLS_PER_ACCOUNT:
        logger.info(f"La cuenta '{account['name']}' ha alcanzado su límite de {MAX_CALLS_PER_ACCOUNT} llamadas.")
        _switch_to_next_twitter_account()
        account = twitter_accounts[current_account_index] # Obtiene la nueva cuenta
        # Si la nueva cuenta también está agotada (y no es la única), se registrará una advertencia
        if account['calls_made'] >= MAX_CALLS_PER_ACCOUNT and len(twitter_accounts) > 1:
            logger.warning(f"La cuenta '{account['name']}' a la que se cambió también está agotada. Esto puede indicar que todas las cuentas están agotadas o que el límite es muy bajo.")

    try:
        # auth_v1 y api_v1 se mantienen por consistencia con el prompt original,
        # aunque client_v2 es el que se usa para la mayoría de las operaciones modernas.
        auth_v1 = tweepy.OAuth1UserHandler(
            account['api_key'], account['api_secret'],
            account['access_token'], account['access_token_secret']
        )
        api_v1 = tweepy.API(auth_v1)
        client_v2 = tweepy.Client(bearer_token=account['bearer_token'])
        logger.info(f"Usando cuenta de API: '{account['name']}'. Llamadas realizadas: {account['calls_made']}")
        return api_v1, client_v2
    except Exception as e:
        logger.error(f"Error de autenticación con la cuenta '{account['name']}': {e}")
        # Si la autenticación falla, se cambia a la siguiente cuenta y se reintenta (manejado por el bucle de reintentos)
        _switch_to_next_twitter_account()
        return None, None
    
def _get_tweets_from_twitter_api(query_or_url: str, max_results: int = 10) -> pd.DataFrame:
    """
    Obtiene tweets de la API de Twitter (X) basándose en una consulta, utilizando
    un sistema de rotación de cuentas para manejar límites de tasa.

    Args:
        query_or_url (str): La consulta de búsqueda (ej. "#hashtag", "@usuario",
                            URL de tweet como "https://x.com/user/status/ID", o ID numérico de tweet).
        max_results (int): Número máximo de resultados a devolver (máx. 100 por llamada para search_recent_tweets).

    Returns:
        pd.DataFrame: Un DataFrame de pandas con los tweets encontrados y sus metadatos,
                      incluyendo columnas 'query' y 'source' para el análisis comparativo.
                      Devuelve un DataFrame con un mensaje de error o vacío si no se encuentran datos.
    """
    if not twitter_accounts:
        logger.error("No hay cuentas de Twitter configuradas. No se pueden obtener tweets.")
        return pd.DataFrame({"Error": ["No hay cuentas de Twitter (X) configuradas."]})

    tweets_list = []
    MAX_RETRIES = len(twitter_accounts) # Intentar con cada cuenta una vez
    retries = 0

    while retries < MAX_RETRIES:
        _api_v1, client_v2 = _get_current_twitter_clients() # Obtiene el cliente de la API con rotación
        if not client_v2:
            logger.error(f"No se pudo obtener un cliente de Twitter (X) válido en el intento {retries + 1}.")
            retries += 1
            time.sleep(1) # Pequeña pausa antes de reintentar
            continue

        try:
            response_data = None
            tweet_fields = ['created_at', 'public_metrics', 'author_id', 'conversation_id', 'in_reply_to_user_id', 'lang']
            expansions = ['author_id', 'in_reply_to_user_id']
            user_fields = ['username', 'name', 'profile_image_url', 'verified']

            # Lógica para diferentes tipos de consulta
            if ("x.com/" in query_or_url or "twitter.com/" in query_or_url) and "/status/" in query_or_url:
                match = re.search(r'.*/status/(\d+)', query_or_url)
                if match:
                    tweet_id = match.group(1)
                    logger.info(f'Buscando tweet por ID: {tweet_id}')
                    response_data = client_v2.get_tweet(id=tweet_id, tweet_fields=tweet_fields, expansions=expansions, user_fields=user_fields)
                    if response_data.data: response_data.data = [response_data.data] # Envuelve el tweet individual en una lista para consistencia
                else:
                    return pd.DataFrame({"Error": ["No se pudo extraer el ID del tweet de la URL."]})
            elif query_or_url.startswith("#"):
                logger.info(f'Buscando tweets por hashtag: {query_or_url}')
                response_data = client_v2.search_recent_tweets(query=query_or_url, tweet_fields=tweet_fields, expansions=expansions, user_fields=user_fields, max_results=min(max_results, 100))
            elif query_or_url.startswith("@"):
                username_to_search = query_or_url.lstrip('@')
                logger.info(f'Buscando tweets del usuario: {username_to_search}')
                #user_lookup = client_v2.get_user(username=username_to_search, user_fields=['id'])
                user_lookup = client_v2.get_user(username=username_to_search, user_fields=['id', 'name', 'username'])
                if user_lookup.data:
                    user_id = user_lookup.data.id
                    #response_data = client_v2.get_users_tweets(id=user_id, tweet_fields=tweet_fields, expansions=expansions, user_fields=user_fields, max_results=min(max_results, 100))
                    response_data = client_v2.get_users_tweets(id=user_id, tweet_fields=tweet_fields, expansions=expansions, user_fields=user_fields, max_results=min(max_results, 100))
                else:
                    return pd.DataFrame({"Error": [f'Usuario de Twitter no encontrado: {username_to_search}']})
            elif query_or_url.isdigit(): # Si es solo un ID de tweet numérico
                tweet_id = query_or_url
                logger.info(f'Buscando tweet por ID numérico: {tweet_id}')
                response_data = client_v2.get_tweet(id=tweet_id, tweet_fields=tweet_fields, expansions=expansions, user_fields=user_fields)
                if response_data.data: response_data.data = [response_data.data] # Consistencia
            else:
                # Este es el caso de búsqueda general (ej. "Elektra", "Coppel")
                logger.info(f"Consulta general detectada: '{query_or_url}'. Intentando buscar usuario similar.")
                # Busca el usuario más relevante basado en la consulta general
                logger.info(f"Paso 1: Buscando usuario exacto '@{query_or_url}' con API v2.")
                #user_lookup_v2 = client_v2.get_user(username=query_or_url, user_fields=user_fields)
                user_lookup_v2 = client_v2.get_user(username=query_or_url, user_fields=user_fields)


                if user_lookup_v2.data:
                    user_id = user_lookup_v2.data.id
                    logger.info(f"Usuario exacto encontrado {query_or_url}")  
                    response_data = client_v2.get_users_tweets(id=user_id, tweet_fields=tweet_fields, expansions=expansions, user_fields=user_fields, max_results=min(max_results, 10))
                else:
                   # Paso 2: Si no hay usuario exacto, buscar usuarios similares (API v1.1 es necesaria para esto).
                    logger.info(f"Paso 2: No se encontró usuario exacto. Buscando usuarios similares con API v1.1.")
                    # La búsqueda de usuarios por un texto general solo está disponible en la API v1.1
                    users_lookup_v1 = _api_v1.search_users(q=query_or_url, count=1)
                    if users_lookup_v1:
                        most_relevant_user = users_lookup_v1[0]
                        user_id = most_relevant_user.id
                        logger.info(f"Usuario más relevante encontrado: '{most_relevant_user.screen_name}' (ID: {user_id}). Obteniendo sus tweets.")
                        response_data = client_v2.get_users_tweets(id=user_id, tweet_fields=tweet_fields, expansions=expansions, user_fields=user_fields, max_results=min(max_results, 10))
                    else:
                        # Paso 3: Si no se encuentra ningún usuario, buscar tweets con el texto.
                        logger.warning(f"Paso 3: No se encontraron usuarios. Buscando tweets que contengan '{query_or_url}'.")
                        response_data = client_v2.search_recent_tweets(query=query_or_url, tweet_fields=tweet_fields, expansions=expansions, user_fields=user_fields, max_results=min(max_results, 10))

            #else:
            #    logger.warning(f"Consulta de Twitter no válida: '{query_or_url}'.")
            #    return pd.DataFrame({"Error": ["Consulta de Twitter no válida (formato @usuario, #hashtag, ID o URL de tweet)"]})

            # Incrementa el contador de llamadas para la cuenta actual
            twitter_accounts[current_account_index]['calls_made'] += 1

            if response_data and response_data.data:
                users_data = {u["id"]: u for u in response_data.includes.get('users', [])} if response_data.includes else {}
                for tweet_obj in response_data.data:
                    author_info = users_data.get(tweet_obj.author_id, {})
                    metrics = tweet_obj.public_metrics if tweet_obj.public_metrics else {}
                    tweets_list.append({
                        'tweet_id': tweet_obj.id, 'text': tweet_obj.text, 'author_id': tweet_obj.author_id,
                        'username': author_info.get("username", "N/A"), 'author_name': author_info.get("name", "N/A"),
                        'author_verified': author_info.get("verified", False), 'created_at': tweet_obj.created_at,
                        'like_count': metrics.get('like_count', 0), 'retweet_count': metrics.get('retweet_count', 0),
                        'reply_count': metrics.get('reply_count', 0), 'quote_count': metrics.get('quote_count', 0),
                        'impression_count': metrics.get('impression_count', 0), 'conversation_id': tweet_obj.conversation_id,
                        'in_reply_to_user_id': tweet_obj.in_reply_to_user_id, 'lang': tweet_obj.lang
                        ,'query': query_or_url, # Añadir la query original para el contexto de comparación
                        'source': 'Twitter' # Añadir la fuente para el análisis comparativo
                    })
                df = pd.DataFrame(tweets_list)
                df['origin'] = "twitter" # Columna 'origin' para identificar la plataforma en análisis posteriores
                return df
            else:
                # Si no hay datos, pero no hay error, significa que no hay resultados para la consulta
                return pd.DataFrame({'Mensaje': [f"No se encontraron tweets para la consulta '{query_or_url}'."]})

        except tweepy.TooManyRequests:
            logger.warning(f"Límite de tasa excedido para la cuenta '{twitter_accounts[current_account_index]['name']}'. Cambiando de cuenta...")
            _switch_to_next_twitter_account()
            retries += 1
            time.sleep(1) # Pequeña pausa antes de reintentar
        except tweepy.TweepyException as e:
            error_message = str(e)
            # Intenta extraer un mensaje de error más específico de la excepción de Tweepy
            if hasattr(e, 'api_errors') and e.api_errors and isinstance(e.api_errors, list) and e.api_errors[0]:
                api_error = e.api_errors[0]
                if isinstance(api_error, dict):
                    error_message = api_error.get('detail', error_message)
                    if 'title' in api_error: error_message = f"{api_error['title']}: {error_message}"
                elif hasattr(api_error, 'message'): error_message = api_error.message
            elif hasattr(e, 'response') and e.response is not None:
                 try: error_details = e.response.json(); error_message = error_details.get('detail', error_details.get('title', str(e)))
                 except ValueError: error_message = e.response.text if e.response.text else str(e)
            logger.error(f"TweepyException para '{query_or_url}' con cuenta '{twitter_accounts[current_account_index]['name']}': {error_message}", exc_info=True)
            _switch_to_next_twitter_account() # Cambia de cuenta también en otros errores de Tweepy
            retries += 1
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error general en _get_tweets_from_twitter_api para '{query_or_url}' con cuenta '{twitter_accounts[current_account_index]['name']}': {e}", exc_info=True)
            _switch_to_next_twitter_account() # Cambia de cuenta en errores generales
            retries += 1
            time.sleep(1)

    logger.error(f"Falló la obtención de tweets para '{query_or_url}' después de {MAX_RETRIES} intentos con todas las cuentas disponibles.")
    return pd.DataFrame({"Error": [f"Falló la obtención de tweets después de múltiples intentos para '{query_or_url}'."]})
