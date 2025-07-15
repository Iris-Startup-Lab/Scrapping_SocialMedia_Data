# data_fetchers.py
import os
import logging
import random
import time
import json
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Selenium
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options as ChromeOptions # Renombrar para evitar conflicto
from selenium_stealth import stealth  
from seleniumwire import webdriver as seleniumwire_webdriver 

# APIs de terceros directas
import googlemaps
import tweepy
from googleapiclient.discovery import build # YouTube
import praw # Reddit
from google_play_scraper import app as play_app, reviews as play_reviews, Sort, reviews_all, search

# Import global constants from config.py
from config import (
    YOUTUBE_API_KEY, MAPS_API_KEY, TWITTER_BEARER_TOKEN,
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    CHROME_DRIVER_PATH, CHROME_BINARY_PATH, PLATFORM_STEALTH, user_agents
)

logger = logging.getLogger(__name__)

# --- Clase FacebookScraper (Solo para Login y Cookies) ---
class FacebookScraper:
    FACEBOOK_LOGIN_URL = 'https://www.facebook.com/login/'
    EMAIL_INPUT_SELECTOR = "input[name='email']"
    PASSWORD_SELECTOR_LOGIN = "input[name='pass']"
    LOGIN_BUTTON_SELECTOR = "button[type='submit']"
    LOGIN_SUCCESS_INDICATOR_XPATH = "//input[@aria-label='Buscar en Facebook'] | //div[@role='feed'] | //div[contains(@aria-label, 'Crear publicación')]"

    COOKIES_ACCEPT_BUTTON_XPATH = '//div[@aria-label="Allow all cookies" and @role="button"]' 
    COOKIES_IFRAME_SELECTOR = None # Rellena esto si descubres un iframe. Ej: "iframe[title='Privacy & Cookies policy']"
    COOKIE_DIALOG_CONTAINER_SELECTOR = "div[role='dialog'][aria-describedby*='dialog_body'], div[data-testid*='cookie-dialog'], div[role='dialog'][aria-label*='cookie']"

    def __init__(self, email: str, password: str, driver_path: str, binary_path: str, wait_time: int = 30, proxy_options: dict = None):
        self.email = email
        self.password = password
        self.driver_path = driver_path
        self.binary_path = binary_path
        self.wait_time = wait_time
        self.driver: seleniumwire_webdriver.Chrome = None
        self.wait: WebDriverWait = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.user_agents = user_agents 
        self.proxy_options = proxy_options 

    def _start_driver(self) -> seleniumwire_webdriver.Chrome: 
        self.logger.info('Iniciando el driver de Chrome...')
        service = Service(self.driver_path)
        options = seleniumwire_webdriver.ChromeOptions()
        options.binary_location = self.binary_path
        
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-notifications")
        options.add_argument('--disable-popup-blocking')
        options.add_argument('--lang=es-MX')
        options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument(f"user-agent={random.choice(self.user_agents)}") 
        
        driver_class = seleniumwire_webdriver.Chrome
        driver_kwargs = {"service": service, "options": options}
        if self.proxy_options:
            driver_kwargs['seleniumwire_options'] = self.proxy_options
        
        try:
            self.driver = driver_class(**driver_kwargs)
            stealth(self.driver,
                    languages=["es-MX", "es"], 
                    vendor="Google Inc.",
                    platform=PLATFORM_STEALTH,
                    webgl_vendor="Intel Inc.",
                    renderer="Intel Iris OpenGL Engine",
                    fix_hairline=True,
            )
            self.logger.info('Selenium-stealth aplicado.')
        except ImportError:
            self.logger.warning('selenium-stealth no está instalado. El scraper podría ser más fácilmente detectado.')
            self.driver = driver_class(**driver_kwargs)
        except Exception as e:
            self.logger.error(f"Error al aplicar selenium-stealth, o al iniciar driver: {e}", exc_info=True)
            self.driver = driver_class(**driver_kwargs)

        self.logger.info('Driver de Chrome iniciado.')
        self.wait = WebDriverWait(self.driver, self.wait_time)
        return self.driver

    def _simulate_human_typing(self, element, text):
        for char in text:
            element.send_keys(char)
            time.sleep(random.uniform(0.05, 0.15))
        time.sleep(random.uniform(0.5, 1.5))

    def _login_to_facebook(self) -> bool:
        self.logger.info('Intentando hacer login en FB.')
        self.driver.get(self.FACEBOOK_LOGIN_URL)
        
        try:
            # --- Paso 1: Intentar cerrar cualquier pop-up de cookies/consentimiento ---
            self.logger.info("Buscando y aceptando cookies si existen...")
            
            original_window = self.driver.current_window_handle
            try:
                if self.COOKIES_IFRAME_SELECTOR:
                    self.logger.info(f"Buscando iframe de cookies con selector: {self.COOKIES_IFRAME_SELECTOR}")
                    iframe_element = self.wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, self.COOKIES_IFRAME_SELECTOR)))
                    self.driver.switch_to.frame(iframe_element)
                    self.logger.info("Cambiado a contexto de iframe de cookies.")
                else:
                    self.logger.info("No se especificó selector de iframe para cookies.")
            except TimeoutException:
                self.logger.info("No se encontró iframe de cookies.")
                self.driver.switch_to.window(original_window)
            except Exception as e:
                self.logger.warning(f"Error al intentar cambiar a iframe de cookies: {e}", exc_info=True)
                self.driver.switch_to.window(original_window)

            try:
                self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, self.COOKIE_DIALOG_CONTAINER_SELECTOR)))
                self.logger.info("Contenedor del diálogo de cookies detectado.")

                accept_cookies_button = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, self.COOKIES_ACCEPT_BUTTON_XPATH)) 
                )
                self.logger.info("Botón 'Allow all cookies' detectado. Clickeando con JavaScript.")
                self.driver.execute_script("arguments[0].click();", accept_cookies_button)
                self.logger.info("Botón de aceptar cookies clickeado exitosamente.")
                time.sleep(random.uniform(3, 6))
                
                if self.COOKIES_IFRAME_SELECTOR: # Volver al contexto principal si cambiamos a iframe
                    self.driver.switch_to.default_content()
                    self.logger.info("Volviendo al contexto principal (después de iframe de cookies).")


            except TimeoutException: 
                self.logger.info('No se encontró el botón de aceptación de cookies dentro del tiempo esperado. Continuando con el login.')
            except Exception as e:
                self.logger.error(f"Error al intentar aceptar cookies: {e}", exc_info=True)
                self.driver.save_screenshot('facebook_login_cookies_fail.png')
                self.logger.warning("Screenshot 'facebook_login_cookies_fail.png' guardado para depuración.")

            # --- Paso 2: Ingresar credenciales ---
            self.logger.info('Iniciando sesión en Facebook...')
            email_input = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, self.EMAIL_INPUT_SELECTOR)))
            self._simulate_human_typing(email_input, self.email)
            
            password_input = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, self.PASSWORD_SELECTOR_LOGIN)))
            self._simulate_human_typing(password_input, self.password)
            self.logger.info('Usuario y contraseña ingresados')
            
            # --- Paso 3: Clic en el botón de login ---
            login_button = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, self.LOGIN_BUTTON_SELECTOR)))
            
            try:
                self.driver.execute_script("arguments[0].click();", login_button)
                self.logger.info("Botón de login clickeado con JavaScript.")
            except Exception as e_click: 
                self.logger.error(f"Error al clickear el botón de login: {e_click}", exc_info=True)
                self.logger.error(f"Página actual al fallar login: {self.driver.current_url}")
                self.driver.save_screenshot('facebook_login_intercepted_error.png')
                self.logger.error("Screenshot 'facebook_login_intercepted_error.png' guardado para depuración.")
                return False 

            # --- Paso 4: Esperar por el éxito del login ---
            self.logger.info("Esperando que el login se procese...")
            self.wait.until(
                EC.url_to_be("https://www.facebook.com/") or 
                EC.presence_of_element_located((By.XPATH, self.LOGIN_SUCCESS_INDICATOR_XPATH))
            )
            self.logger.info("Login exitoso.")
            return True
        except TimeoutException as e:
            logger.error(f"Timeout esperando elementos o login. Login falló o estructura de la página cambió: {e}", exc_info=True)
            logger.error(f"Página actual al fallar login: {self.driver.current_url}")
            self.driver.save_screenshot('facebook_login_timeout_fail.png')
            logger.error("Screenshot 'facebook_login_timeout_fail.png' guardado.")
            return False
        except Exception as e:
            logger.error(f"Error inesperado durante el login: {e}", exc_info=True)
            logger.error(f"Página actual al fallar login: {self.driver.current_url}")
            self.driver.save_screenshot('facebook_login_general_fail.png')
            return False

    def get_current_cookies(self) -> list[dict]: 
        """
        Obtiene todas las cookies de la sesión actual del driver.
        Debe ser llamado DESPUÉS de un login exitoso.
        """
        if self.driver:
            self.logger.info("Obteniendo cookies de la sesión actual...")
            cookies = self.driver.get_cookies()
            return cookies
        self.logger.warning("No hay driver iniciado para obtener cookies.")
        return []

    def close_driver(self):
        if self.driver:
            self.driver.quit()
            self.logger.info("Driver de Chrome cerrado.")
            self.driver = None
            self.wait = None

# --- Funciones de Scraping de Contenido por Plataforma ---
# Estas funciones DEBEN devolver un DataFrame con una columna 'origin'
# y una columna de texto consistente ('text', 'comment', o 'content').

def _make_oxylabs_request(url: str, cookies_string: str, max_results: int, purpose: str) -> dict:
    from config import OXYLABS_SCRAPER_API_ENDPOINT, OXYLABS_API_USER, OXYLABS_API_PASS
    
    if not all([OXYLABS_API_USER, OXYLABS_API_PASS]):
        logger.error("ERROR: Credenciales de Oxylabs Scraper API no configuradas en variables de entorno.")
        return {"error": "Credenciales de Oxylabs Scraper API no configuradas."}

    payload = {
        "source": "facebook", 
        "url": url,
        "geo": "US", 
        "locale": "en-US", 
        "render": "html", 
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        "cookies": cookies_string, 
        "limit": max_results, 
    }

    headers = {
        "Content-Type": "application/json",
    }

    try:
        logger.info(f"Oxylabs API: Enviando solicitud para {purpose} a {url}...")
        response = requests.post(
            OXYLABS_SCRAPER_API_ENDPOINT, 
            headers=headers, 
            data=json.dumps(payload),
            auth=(OXYLABS_API_USER, OXYLABS_API_PASS)
        )
        response.raise_for_status()
        result_data = response.json()
        
        if result_data and result_data.get("results"):
            return result_data["results"][0]
        else:
            logger.warning(f"Oxylabs API: Respuesta vacía o sin resultados para {purpose}. URL: {url}")
            return {"error": f"Oxylabs API no devolvió contenido para {purpose}."}

    except requests.exceptions.RequestException as e:
        logger.error(f"Oxylabs API: Error de solicitud HTTP para {purpose}: {e}", exc_info=True)
        return {"error": f"Error de red o API de Oxylabs para {purpose}: {e}"}
    except Exception as e:
        logger.error(f"Oxylabs API: Error inesperado al procesar la respuesta para {purpose}: {e}", exc_info=True)
        return {"error": f"Error al procesar respuesta de Oxylabs para {purpose}: {e}"}

def _scrape_facebook_posts_comments_oxylabs(search_term: str, cookies_string: str, max_posts_to_scrape: int = 1, max_comments_per_post: int = 10) -> pd.DataFrame:
    """
    Realiza una búsqueda de posts en Facebook y scrapea sus comentarios
    usando la API de Oxylabs Scraper con cookies pre-obtenidas.
    """
    all_comments_df = pd.DataFrame()
    
    if not cookies_string:
        logger.error("ERROR: No se proporcionó la cadena de cookies para Oxylabs Facebook Scraper.")
        return pd.DataFrame({'Error': ["No se proporcionaron cookies."]})

    # FASE 1: Scrapear la página de resultados de búsqueda de Facebook
    search_url = f"https://www.facebook.com/search/posts?q={search_term}"
    logger.info(f"FASE 1: Buscando posts para '{search_term}' en Facebook via Oxylabs...")
    
    search_results_oxylabs_response = _make_oxylabs_request(
        url=search_url, 
        cookies_string=cookies_string, 
        max_results=max_posts_to_scrape, 
        purpose="búsqueda de posts"
    )

    if "error" in search_results_oxylabs_response:
        return pd.DataFrame({'Error': [search_results_oxylabs_response["error"]]})
    
    scraped_html = search_results_oxylabs_response.get("content")
    if not scraped_html:
        return pd.DataFrame({'Mensaje': [f"Oxylabs no devolvió HTML para la búsqueda de '{search_term}'."]})

    # FASE 2: Parsear el HTML de búsqueda para obtener URLs de posteos
    logger.info("FASE 2: Parseando HTML de resultados de búsqueda para extraer URLs de posts...")
    soup = BeautifulSoup(scraped_html, 'html.parser')
    post_links = []
    
    # --- CRÍTICO: AJUSTAR ESTOS SELECTORES para el HTML que Oxylabs te devuelve ---
    # Estos selectores son para el HTML directo de Facebook. Oxylabs puede servir un HTML ligeramente diferente.
    # Necesitas inspeccionar el 'oxylabs_response_busqueda_de_posts.html' que guardaste para encontrar los selectores de los enlaces a los posts.
    # Sugerencias: busca por div[role="article"] o div[data-pagelet*="FeedUnit"] y luego links dentro.
    
    # Este es un selector muy genérico, puede que necesites algo más específico de Facebook.
    # Revisa los atributos data-testid si existen, son más estables.
    # Ejemplo de selector basado en el HTML común de Facebook posts.
    post_link_elements = soup.find_all(lambda tag: tag.name == 'a' and
                                     'href' in tag.attrs and
                                     ('/posts/' in tag['href'] or '/videos/' in tag['href'] or '/photo/' in tag['href']) and
                                     ('fbid=' in tag['href'] or 'story_fbid=' in tag['href']) and
                                     'comment_id' not in tag['href'] # Evitar links de comentarios
                                    )
    
    for el in post_link_elements:
        href = el.get('href')
        if href and 'facebook.com' in href: # Filtrar por dominio de Facebook
            parsed_url = href.split('?')[0].split('&')[0] # Limpiar URL de parámetros de seguimiento
            post_links.append(parsed_url)
            
    unique_post_links = list(dict.fromkeys(post_links))
    logger.info(f"FASE 2: Encontrados {len(unique_post_links)} URLs de posts únicas.")

    if not unique_post_links:
        return pd.DataFrame({'Mensaje': [f"No se encontraron URLs de posts en la página de búsqueda para '{search_term}'."]})

    posts_to_scrape_comments_from = unique_post_links[:max_posts_to_scrape]
    list_of_comments_dfs = []

    # FASE 3: Scrapear y Parsear COMENTARIOS de cada post
    for i, post_url in enumerate(posts_to_scrape_comments_from):
        logger.info(f"FASE 3: Scrapeando comentarios del post {i+1}/{len(posts_to_scrape_comments_from)}: {post_url}...")
        post_oxylabs_response = _make_oxylabs_request(
            url=post_url, 
            cookies_string=cookies_string, 
            max_results=max_comments_per_post, 
            purpose=f"comentarios del post {post_url}"
        )

        if "error" in post_oxylabs_response:
            logger.error(f"Error al obtener comentarios del post {post_url}: {post_oxylabs_response['error']}")
            list_of_comments_dfs.append(pd.DataFrame({'Error': [f"Error al obtener comentarios de {post_url}: {post_oxylabs_response['error']}"]}))
            continue
        
        post_html_content = post_oxylabs_response.get("content")
        if not post_html_content:
            logger.warning(f"Oxylabs no devolvió HTML para comentarios del post {post_url}.")
            list_of_comments_dfs.append(pd.DataFrame({'Mensaje': [f"No hay HTML para comentarios de {post_url}."]}))
            continue
        
        # --- CRÍTICO: AJUSTAR ESTOS SELECTORES PARA COMENTARIOS del HTML de Oxylabs ---
        # Necesitas inspeccionar el 'oxylabs_response_comentarios_del_post.html' que guardaste.
        soup_post = BeautifulSoup(post_html_content, 'html.parser')
        current_post_comments = []
        
        # Selector genérico para contenedores de comentarios (¡AJUSTAR!)
        # Sugerencia: Busca divs con `role='article'` que tengan `aria-label` que contenga 'comment'.
        comment_divs = soup_post.find_all('div', {'role': 'article', 'aria-label': lambda x: x and 'comment' in x.lower()})
        
        if not comment_divs:
            logger.warning(f"No se encontraron contenedores de comentarios en el HTML de Oxylabs para {post_url}.")
            list_of_comments_dfs.append(pd.DataFrame({'Mensaje': [f"No se encontraron comentarios para {post_url}."]}))
            continue

        for comment_container in comment_divs:
            try:
                # --- SELECTORES ESPECÍFICOS DENTRO DEL CONTENEDOR DE COMENTARIOS (¡AJUSTAR!) ---
                # Inspecciona el HTML de la respuesta de Oxylabs.
                username_el = comment_container.find('span', class_=lambda x: x and 'x193iq5w' in x and 'x1lliihq' in x) 
                comment_text_el = comment_container.find('div', {'dir': 'auto', 'style': True}) # Buscar div con dir='auto' y algún estilo
                timestamp_el = comment_container.find('a', {'role': 'link', 'href': True, 'tabindex': '-1'})
                
                username = username_el.get_text(strip=True) if username_el else "N/A"
                comment_text = comment_text_el.get_text(strip=True) if comment_text_el else "N/A"
                timestamp_text = timestamp_el.get_text(strip=True) if timestamp_el else "N/A" 
                
                current_post_comments.append({
                    'post_url': post_url,
                    'username': username,
                    'comment_text': comment_text,
                    'timestamp': timestamp_text,
                    'origin': 'facebook', # Establecer el origen aquí
                })
            except Exception as e:
                logger.warning(f"Error al extraer detalles de un comentario en {post_url}: {e}. Salto el comentario.")
                continue
        
        if current_post_comments:
            list_of_comments_dfs.append(pd.DataFrame(current_post_comments))
        else:
            logger.info(f"No se pudieron extraer comentarios detallados del post {post_url}.")

    if list_of_comments_dfs:
        all_comments_df = pd.concat(list_of_comments_dfs, ignore_index=True)
        logger.info(f"FASE 4: Total de comentarios extraídos para '{search_term}': {len(all_comments_df)}.")
    else:
        all_comments_df = pd.DataFrame({'Mensaje': [f"No se extrajo ningún comentario para la búsqueda '{search_term}'."]})
        
    return all_comments_df


# --- Funciones de Extracción de Datos por Plataforma (AHORA CON ARGUMENTOS) ---
# TODAS ESTAS FUNCIONES DEBEN DEVOLVER UN DATAFRAME CON UNA COLUMNA 'origin'
# Y una columna de texto consistente ('text', 'comment', o 'content').

def _get_tweets_from_twitter_api(query_or_url: str) -> pd.DataFrame:
    if not TWITTER_BEARER_TOKEN:
        return pd.DataFrame({"Error": ["Bearer Token de Twitter no configurado."]})

    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
    tweets_list = []
    response_data = None

    tweet_fields = ['created_at', 'public_metrics', 'author_id', 'conversation_id', 'in_reply_to_user_id', 'lang']
    expansions = ['author_id', 'in_reply_to_user_id']
    user_fields = ['username', 'name', 'profile_image_url', 'verified']
    max_results_count = 10

    try:
        # Check if it's a tweet URL or ID
        if ("x.com/" in query_or_url or "twitter.com/" in query_or_url) and "/status/" in query_or_url:
            match = re.search(r'.*/status/(\d+)', query_or_url)
            if match:
                tweet_id = match.group(1)
                logger.info(f'Buscando tweet por ID: {tweet_id}')
                response_data = client.get_tweet(id=tweet_id, tweet_fields=tweet_fields, expansions=expansions, user_fields=user_fields)
                if response_data.data: response_data.data = [response_data.data] # Wrap single tweet in list
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
        elif query_or_url.isdigit(): # If it's just a numeric tweet ID
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
                })
        
        if not tweets_list: return pd.DataFrame({'Mensaje': [f"No se encontraron tweets para la consulta '{query_or_url}' o la respuesta no contenía datos."]})
        df = pd.DataFrame(tweets_list)
        df['origin'] = "twitter" 
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

def _get_youtube_channels_and_comments(query: str, max_results_channels: int = 5, max_results_videos: int = 20, max_results_comments: int = 10) -> pd.DataFrame:
    """
    Busca canales de YouTube por nombre, obtiene videos de los canales similares, y luego sus comentarios.
    """
    if not YOUTUBE_API_KEY:
        return pd.DataFrame({'Error': ["Clave API de YouTube no configurada."]})
    
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    
    def _search_youtube_channels(youtube_service, query_term, max_res):
        request = youtube_service.search().list(q=query_term, type='channel', part='snippet', maxResults=max_res)
        response = request.execute()
        channels = []
        for item in response.get('items', []):
            channels.append({'channelId': item['id']['channelId'], 'title': item['snippet']['title'], 'description': item['snippet']['description']})
        return channels
    
    def _search_videos_from_channel(channel_id, youtube_service, max_res):
        request = youtube_service.search().list(channelId=channel_id, part='snippet', maxResults=max_res, order='date')
        response = request.execute()
        videos = []
        for item in response.get('items', []):
            if item['id']['videoId']: # Ensure it's a video item
                videos.append({'videoId': item['id']['videoId'], 'title': item['snippet']['title']})
        return videos

    def _get_comments_from_video(video_id, youtube_service, max_res):
        comments = []
        try:
            request = youtube_service.commentThreads().list(part='snippet', videoId=video_id, textFormat='plainText', maxResults=max_res)
            response = request.execute()
            for item in response.get('items', []):
                comment_snippet = item['snippet']['topLevelComment']['snippet']
                comments.append({'author': comment_snippet['authorDisplayName'], 'comment': comment_snippet['textDisplay'], 'publishedAt': comment_snippet['publishedAt']})
        except Exception as e:
            logger.warning(f"Error fetching comments for video {video_id}: {e}")
        return comments

    try:
        logger.info(f"Buscando canales de YouTube similares a '{query}'...")
        busqueda_canales = _search_youtube_channels(youtube, query, max_results_channels)
        
        canales_similares = []
        for canal in busqueda_canales:
            channel_title_search = canal.get('title', '').lower()
            if re.search(re.escape(query.lower()), channel_title_search, re.IGNORECASE):
                canales_similares.append(canal)
        
        if not canales_similares:
            logger.info(f"No se encontraron canales similares a '{query}'.")
            return pd.DataFrame({'Mensaje': [f"No se encontraron canales de YouTube para la búsqueda '{query}'."]})

        all_comments_list = []
        for channel_info in canales_similares:
            channel_id = channel_info.get('channelId')
            channel_title = channel_info.get('title')
            
            logger.info(f"Obteniendo videos del canal '{channel_title}' ({channel_id})...")
            busqueda_videos = _search_videos_from_channel(channel_id, youtube, max_results_videos)
            
            for video in busqueda_videos:
                video_id = video.get('videoId')
                video_title = video.get('title')
                if video_id:
                    logger.info(f"Obteniendo comentarios del video '{video_title}' ({video_id})...")
                    comments_from_video = _get_comments_from_video(video_id, youtube, max_results_comments)
                    if comments_from_video:
                        df_comments = pd.DataFrame(comments_from_video)
                        df_comments['video_id'] = video_id
                        df_comments['video_title'] = video_title
                        df_comments['channel_id'] = channel_id
                        df_comments['channel_title'] = channel_title
                        df_comments['query'] = query # Añadir la query original
                        all_comments_list.append(df_comments)
                else:
                    logger.warning(f"No se pudo obtener el ID del video para un video en el canal {channel_title}.")
            
        if all_comments_list:
            comments_df = pd.concat(all_comments_list, ignore_index=True)
            comments_df['origin'] = "youtube"
            return comments_df
        else:
            return pd.DataFrame({'Mensaje': [f"No se encontraron comentarios para la búsqueda '{query}' en YouTube."]})
        
    except Exception as e:
        logger.error(f"Error general al obtener comentarios de YouTube para '{query}': {e}", exc_info=True)
        return pd.DataFrame({"Error": [f"Error general al obtener comentarios de YouTube: {e}"]})


def _get_youtube_comments(video_url_or_id: str) -> pd.DataFrame:
    if not YOUTUBE_API_KEY:
        return pd.DataFrame({'Error': ["Clave API de YouTube no configurada."]})
    
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    video_id = None
    
    if "v=" in video_url_or_id: video_id = video_url_or_id.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in video_url_or_id: video_id = video_url_or_id.split("youtu.be/")[-1].split("?")[0]
    elif len(video_url_or_id) == 11 and video_url_or_id.isalnum(): video_id = video_url_or_id
    
    if not video_id: return pd.DataFrame({'Error': ["URL/ID de YouTube no válido."]})

    try:
        comments = []
        response = youtube.commentThreads().list(part='snippet',  videoId=video_id, textFormat='plainText', maxResults=10).execute()
        
        for item in response.get('items', []):
            comment_snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'author': comment_snippet['authorDisplayName'], 
                'comment': comment_snippet['textDisplay'],
                'video_id': video_id, 'published_at': comment_snippet['publishedAt']
            })
        
        if not comments: return pd.DataFrame({'Mensaje': [f"No se encontraron comentarios para el video de YouTube {video_id}."]})

        df = pd.DataFrame(comments)
        df['origin'] = "youtube"
        return df
    except Exception as e:
        logger.error(f"Error al obtener comentarios de YouTube para '{video_url_or_id}': {e}", exc_info=True)
        return pd.DataFrame({"Error": [f"Error al obtener comentarios de YouTube: {e}"]})

def _maps_comments(maps_query_or_url: str) -> pd.DataFrame:
    if not MAPS_API_KEY: return pd.DataFrame({'Error': ["Clave API de Google Maps no configurada."]})
    
    gmaps = googlemaps.Client(MAPS_API_KEY)
    comments = []
    
    place_id = None
    if maps_query_or_url.startswith("http") and "place_id" in maps_query_or_url:
        match = re.search(r'place_id=([^&]+)', maps_query_or_url)
        if match: place_id = match.group(1)
    elif maps_query_or_url.startswith("http") and "@" in maps_query_or_url: 
        logger.warning("Google Maps API no permite obtener reviews directamente de coordenadas en URL.")
        return pd.DataFrame({'Error': ["No se pueden obtener reviews directamente de URLs con coordenadas. Intente un nombre de lugar o URL con place_id."]})
    else: 
        try:
            find_place_result = gmaps.find_place(maps_query_or_url, input_type='textquery')
            if find_place_result['status'] == 'OK' and find_place_result.get('candidates'):
                place_id = find_place_result['candidates'][0]['place_id']
            else: return pd.DataFrame({'Mensaje': [f"No se encontró el lugar para '{maps_query_or_url}' en Google Maps."]})
        except Exception as e:
            logger.error(f"Error al buscar Place ID para '{maps_query_or_url}': {e}", exc_info=True)
            return pd.DataFrame({'Error': [f"Error al buscar lugar en Google Maps: {e}"]})

    if not place_id: return pd.DataFrame({'Error': ["No se pudo determinar un Place ID válido."]})

    try:
        place_details = gmaps.place(place_id, fields=['name', 'rating', 'review'], language='es')
        reviews_data = place_details.get('result', {}).get('reviews', [])

        if reviews_data:
            for review in reviews_data:
                comments.append({
                    'author': review.get('author_name', 'N/A'),  
                    'comment': review.get('text', ''),  
                    'rating': review.get('rating', 'N/A'),
                    'place_name': place_details['result'].get('name', 'N/A'),
                    'origin': 'maps'
                })
        
        if not comments: return pd.DataFrame({"Mensaje": [f"No se encontraron reviews para el lugar '{maps_query_or_url}'."]})

        df = pd.DataFrame(comments)
        return df
    except Exception as e:
        logger.error(f"Error al obtener reviews de Google Maps para '{maps_query_or_url}': {e}", exc_info=True)
        return pd.DataFrame({"Error": [f"Error al obtener reviews de Google Maps: {e}"]})

def _get_reddit_comments(submission_url: str) -> pd.DataFrame:
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        return pd.DataFrame({"Error": ["Credenciales de Reddit API no configuradas."]})

    try:
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT, read_only=True)
        match = re.search(r"comments/([a-zA-Z0-9]+)/?", submission_url)
        if not match: return pd.DataFrame({"Error": ["Formato de URL de Reddit no válido."]})
        
        submission_id = match.group(1)
        submission = reddit.submission(id=submission_id)
        submission.comments.replace_more(limit=None)

        comments_data = []
        for comment in submission.comments.list():
            if isinstance(comment, praw.models.Comment):
                comments_data.append({
                    'id': comment.id, 'original_url': submission_url, 'reddit_title': submission.title, 
                    'author': str(comment.author) if comment.author else "[deleted]",
                    'comment': comment.body, 'score': comment.score,
                    'created_utc': pd.to_datetime(comment.created_utc, unit='s'),
                    'parent_id': comment.parent_id, 'permalink': f"https://www.reddit.com{comment.permalink}",
                    'is_submitter': comment.is_submitter, 'edited': False if isinstance(comment.edited, bool) else pd.to_datetime(comment.edited, unit='s'),
                    'depth': comment.depth, 'origin': 'reddit'
                })
        
        if not comments_data: return pd.DataFrame({"Mensaje": ["No comments found for this submission or comments are not public."]})

        df = pd.DataFrame(comments_data)
        return df
    except praw.exceptions.PRAWException as e:
        logger.error(f"Error de API de PRAW (Reddit) para '{submission_url}': {e}", exc_info=True)
        return pd.DataFrame({"Error": [f"Error de API de Reddit: {e}"]})
    except requests.exceptions.RequestException as e: 
        logger.error(f"Error de red al contactar Reddit para '{submission_url}': {e}", exc_info=True)
        return pd.DataFrame({"Error": [f"Network error while contacting Reddit: {e}"]})
    except Exception as e:
        logger.error(f"Error inesperado al obtener comentarios de Reddit para '{submission_url}': {e}", exc_info=True)
        return pd.DataFrame({"Error": [f"Error al obtener comentarios de Reddit: {e}"]})

def _get_playstore_comments(app_id_or_url: str) -> pd.DataFrame:
    app_id = None
    if re.match(r'.*id=([a-zA-Z0-9\._-]+)', app_id_or_url): app_id = re.search(r'(?<=id=)[^&]+', app_id_or_url).group(0)
    elif re.match(r'^[a-zA-Z0-9\._-]+$', app_id_or_url): app_id = app_id_or_url
    else: return pd.DataFrame({"Error": ["ID/URL de Play Store no válida."]})

    if not app_id: return pd.DataFrame({"Error": ["No se pudo extraer el ID de la aplicación de Play Store."]})

    try:
        reviews, continuation_token = play_reviews(app_id, lang='es', country='mx', sort=Sort.NEWEST, count=30, filter_score_with=None)
        
        if not reviews: return pd.DataFrame({"Mensaje": [f"No se encontraron reviews para la app '{app_id}'."]})

        df = pd.DataFrame(reviews)
        df['origin'] = "playstore"
        df['content'] = df['content'].astype(str)
        return df
    except Exception as e:
        logger.error(f"Error al obtener reviews de Play Store para '{app_id_or_url}': {e}", exc_info=True)
        return pd.DataFrame({"Error": [f"Error al obtener reviews de Play Store: {e}"]})

def _process_wikipedia_for_df(url: str) -> pd.DataFrame:
    if not url: return pd.DataFrame({'Error': ["Por favor ingresa una URL de Wikipedia"]})
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        
        if not paragraphs or (len(paragraphs)==1 and not paragraphs[0].strip()):
            all_text = soup.get_text(separator='\n', strip=True)
            if all_text: paragraphs=[line for line in all_text.splitlines() if line.strip()]
            if not paragraphs: return pd.DataFrame({"Error": ["No se pudo encontrar texto en la URL proporcionada"]})

        df = pd.DataFrame({'paragraph_number': range(1, len(paragraphs) + 1), 'text': paragraphs, 'length': [len(p) for p in paragraphs]})
        df['text'] = df['text'].astype(str)
        df = df[~(df['text'].str.startswith("Error al acceder a Wikipedia:") | df['text'].str.strip() == "")]
        
        if df.empty: return pd.DataFrame({"Error": ["No se encontraron párrafos válidos en la URL proporcionada"]})
        
        df['origin'] = "wikipedia"
        return df
    except Exception as e:
        logger.error(f"Error al obtener texto de Wikipedia para '{url}': {e}", exc_info=True)
        return pd.DataFrame({"Error": [f"Error al acceder a Wikipedia: {e}"]})

def _process_generic_webpage_for_df(url: str) -> pd.DataFrame:
    if not url: return pd.DataFrame({'Error': ["Por favor ingresa una URL"]})
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script_or_style in soup(["script", "style"]): script_or_style.extract()
            
        paragraphs_elements = soup.find_all('p')
        paragraphs = [p.get_text(strip=True) for p in paragraphs_elements]
        
        if not paragraphs or (len(paragraphs) == 1 and not paragraphs[0].strip()):
            all_text = soup.get_text(separator='\n', strip=True)
            if all_text: paragraphs = [line for line in all_text.splitlines() if line.strip()]
            if not paragraphs: return pd.DataFrame({"Error": ["No se pudo encontrar texto"]})
        
        df = pd.DataFrame({'paragraph_number': range(1, len(paragraphs) + 1), 'text': paragraphs, 'length': [len(p) for p in paragraphs]})
        df['text'] = df['text'].astype(str)
        df['origin'] = "generic_webpage"
        return df
    except Exception as e:
        logger.error(f"Error al acceder a la página web '{url}': {e}", exc_info=True)
        return pd.DataFrame({"Error": [f"Error al acceder a la página web: {e}"]})
