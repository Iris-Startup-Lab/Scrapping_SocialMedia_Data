# -*- coding: utf-8 -*-
#-------------------------------------------------------------
######### Social Media Downloader Shiny App ######
######### VERSION 0.5 ######
#-------------------------------------------------------------


#"cd .\Local\scripts\Social_media_comments\shiny_app\iris_social_media_downloader"
#### Add Pinecone and the button of comments
#### Change the google.generative api to the newest

##### Importando librerías
import os 
### Para directorios
from pathlib import Path

print("--- Environment Variable Check for Cache Config ---")
print(f"Initial os.environ.get('HF_SPACE_ID'): {os.environ.get('HF_SPACE_ID')}")
print(f"Initial os.environ.get('HOME'): {os.environ.get('HOME')}")
print(f"Initial os.environ.get('USER'): {os.environ.get('USER')}")
print(f"Initial os.environ.get('XDG_CACHE_HOME'): {os.environ.get('XDG_CACHE_HOME')}")
print("-------------------------------------------------")

try:
    # Para el servidor, detectando espacios de Hugging Face
    hf_space_id_value = os.environ.get('HF_SPACE_ID')
    is_huggingface_spaces_by_id = bool(hf_space_id_value)
    
    current_home_dir_str = os.path.expanduser('~') 
    current_home_dir = Path(current_home_dir_str)
    
    is_root_home = (current_home_dir_str == "/")

    print(f"DEBUG: HF_SPACE_ID raw value: '{hf_space_id_value}', is_huggingface_spaces_by_id: {is_huggingface_spaces_by_id}")
    print(f"DEBUG: os.path.expanduser('~') resolved to: {current_home_dir_str}, is_root_home: {is_root_home}")
    
    tmp_dir = Path("/tmp")
    tmp_exists = tmp_dir.exists()
    tmp_writable = os.access(str(tmp_dir), os.W_OK) if tmp_exists else False
    print(f"DEBUG: /tmp exists: {tmp_exists}, /tmp writable: {tmp_writable}")

    if is_huggingface_spaces_by_id:
        base_cache_path = tmp_dir / "iris_social_media_downloader_cache"
        print(f"INFO: Detected Hugging Face Spaces environment (by HF_SPACE_ID). Using /tmp for cache. Base path: {base_cache_path}")
    elif is_root_home and tmp_exists and tmp_writable:
        base_cache_path = tmp_dir / "iris_social_media_downloader_cache"
        print(f"INFO: Detected container-like environment (home is '/' and /tmp is writable). Using /tmp for cache. Base path: {base_cache_path}")
    else:
        can_write_to_home_cache = False
        if current_home_dir_str != "/":
            try:
                home_cache_test_path = current_home_dir / ".cache" / "_test_writability"
                os.makedirs(home_cache_test_path, exist_ok=True)
                os.rmdir(home_cache_test_path) 
                can_write_to_home_cache = True
            except OSError:
                can_write_to_home_cache = False
        
        if can_write_to_home_cache:
            base_cache_path = current_home_dir / ".cache" / "iris_social_media_downloader_cache"
            print(f"INFO: Detected standard local environment. Using home-based .cache: {base_cache_path}")
        else:
            script_dir_cache = Path(__file__).resolve().parent / ".app_cache" 
            base_cache_path = script_dir_cache / "iris_social_media_downloader_cache"
            print(f"INFO: Home dir ('{current_home_dir_str}') not suitable for .cache or /tmp fallback failed. Using script-relative cache: {base_cache_path}")

    os.makedirs(base_cache_path, exist_ok=True)
    print(f"DEBUG: Ensured base_cache_path exists: {base_cache_path}")
    
    hf_cache_path = base_cache_path / "huggingface"
    os.environ['HF_HOME'] = str(hf_cache_path)
    print(f"DEBUG: Setting HF_HOME to: {hf_cache_path}")

    mpl_cache_path = base_cache_path / "matplotlib"
    os.environ['MPLCONFIGDIR'] = str(mpl_cache_path)
    print(f"DEBUG: Setting MPLCONFIGDIR to: {mpl_cache_path}")


    os.environ['XDG_CACHE_HOME'] = str(base_cache_path)
    print(f"DEBUG: Setting XDG_CACHE_HOME to: {base_cache_path}")
    
    wdm_driver_cache = base_cache_path / "selenium" 
    os.environ['WDM_DRIVER_CACHE_PATH'] = str(wdm_driver_cache)
    print(f"DEBUG: Setting WDM_DRIVER_CACHE_PATH to: {wdm_driver_cache}")

    wdm_general_cache = base_cache_path / "webdriver_manager" 
    os.environ['WDM_LOCAL'] = str(wdm_general_cache)
    print(f"DEBUG: Setting WDM_LOCAL to: {wdm_general_cache}")

    os.makedirs(hf_cache_path, exist_ok=True)
    os.makedirs(mpl_cache_path, exist_ok=True)
    os.makedirs(wdm_driver_cache, exist_ok=True) 
    os.makedirs(wdm_general_cache, exist_ok=True)
    
    print(f"INFO: Final Cache directory base set to: {base_cache_path}")
    print(f"INFO: Final HF_HOME set to: {os.environ.get('HF_HOME')}")
    print(f"INFO: Final MPLCONFIGDIR set to: {os.environ.get('MPLCONFIGDIR')}")
    print(f"INFO: Final XDG_CACHE_HOME set to: {os.environ.get('XDG_CACHE_HOME')}")
    print(f"INFO: Final WDM_DRIVER_CACHE_PATH set to: {os.environ.get('WDM_DRIVER_CACHE_PATH')}")
    print(f"INFO: Final WDM_LOCAL set to: {os.environ.get('WDM_LOCAL')}")
except Exception as e: 
    print(f"CRITICAL WARNING: An unexpected error occurred during cache setup: {e}")
    import traceback
    traceback.print_exc()
    print("Proceeding without custom cache paths. This will likely lead to errors.")

### Librerías principales de Shiny y relacionadas
from shiny import App, render, ui, reactive
import shinyswatch
import asyncio
#### Librerías para usos diversos
import io
import requests
import time
from dotenv import load_dotenv
import re 
import random # Import random for User-Agent selection
import tempfile
import praw
from scipy.special import softmax
import ast 
from typing import Any, Dict, List as TypingList 
import json 
from functools import partial 
#from shared import app_dir, tips
#### Librerías para análisis de datos y llamado de datos
import numpy as np 
import pandas as pd
#### Librerías para gráficos
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget, render_plotly # xui import removed
from plotly.subplots import make_subplots
from PIL import Image 
from pyvis.network import Network


#### Librerías para web scraping 
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from google_play_scraper import app as play_app, reviews as play_reviews, Sort, reviews_all, search


#### Librerías para uso de las API's oficiales
import googlemaps
import tweepy
from googleapiclient.discovery import build ## Youtube

#### Librerías para análisis de textos 
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from pysentimiento import create_analyzer


#### Librerías para llamar a modelos de HuggingFace
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

#### Librerías para usar LLM's y agentes
import google.generativeai as genai
from openai import OpenAI 
#from chatlas import ChatGoogle, ChatOpenAI
from langchain_openai import ChatOpenAI as LangchainChatOpenAI 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.messages import HumanMessage
from smolagents import CodeAgent, OpenAIServerModel, Tool
from smolagents.utils import make_image_url, encode_image_base64 


#### Librerías para usos específicos
from deep_translator import GoogleTranslator, single_detection ## Traducción y detección de lenguaje

#### Lirerías de Logs
import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#### Librerías para bases de datos
from pinecone import Pinecone, ServerlessSpec


#from crawl4ai import WebCrawler 
## Notas de bugs 
## Funciona todo menos platform inputs


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


print(f'Este es el sistema ooerativo: {os.name}')

if os.name == 'nt':  # Windows
    CHROME_DRIVER_PATH  = os.getenv("CHROME_DRIVER_PATH_WINDOWS")
    print(f"Este es el directorio para Chrome: {CHROME_DRIVER_PATH}")
elif os.name == 'posix':  # Linux o macOS
    CHROME_DRIVER_PATH  = os.getenv("CHROME_DRIVER_PATH_LINUX")
    print(f"Este es el directorio para Chrome: {CHROME_DRIVER_PATH}")
else: 
    CHROME_DRIVER_PATH  = 'user/bin/chromedriver' 
    print(f"Este es el directorio para Chrome: {CHROME_DRIVER_PATH}")



# List of common User-Agents to rotate
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
]

#logo_path = os.path.join(os.path.dirname(__file__), "www", "LogoNuevo.png")
here = Path(__file__).parent

DOMINIOS_PERMITIDOS = ['elektra.com.mx', 'dialogus.com.mx', 'tecnologiaaccionable.mx']

#### Comenzando la UI/Frontend
app_ui = ui.page_fixed(
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="styles.css"), # Your existing stylesheet
        ui.tags.link( # Agregando esto para font awesome y los íconos
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" # Or a specific version bundled with your theme
        ),
        ui.tags.script("""
            $(document).ready(function(){
                $('#gemini_prompt').on('keypress', function(e){
                    // 13 es la clave para Enter
                    if(e.which == 13 && !e.shiftKey){
                        e.preventDefault(); // Prevent default Enter behavior (like adding a new line)
                        $('#ask_gemini').click(); // Trigger the button click
                    }
                });
            });            
        """)
    ),
    #ui.tags.link(rel="stylesheet", href="styles.css"),
    #### """ui antiguo"""
    # ui.layout_sidebar(
    #     ui.sidebar(
    #         #ui.img(src="LogoNuevo.png", style="height: 40px; width: auto; display: block; margin-left: auto; margin-right: auto; margin-bottom: 10px;"),
    #         #ui.output_image("app_logo", width='100px', height='50px'),
    #         #ui.img(src="LogoNuevo.png", height='100px', class_="center-block"),
    #         #ui.img(src="./www/LogoNuevo.png", height='100px', class_="center-block"),
    #         #ui.img(src="E:/Users/1167486/Local/scripts/Social_media_comments/shiny_app/iris_social_media_downloader/www/LogoNuevo.png", height='100px', class_="center-block"),            
    #         ui.markdown("**Social Media Downloader** - Extrae y analiza datos de diferentes plataformas."),
    #         ui.hr(),
    #         # Selector de plataforma
    #         ui.input_select(
    #             "platform_selector",
    #             "Seleccionar Plataforma:",
    #             {
    #                 "wikipedia": "Wikipedia",
    #                 "playstore": "Google Play Store",
    #                 "youtube": "YouTube",
    #                 "maps": "Google Maps",
    #                 "reddit": "Reddit", 
    #                 "twitter": "Twitter (X)",
    #                 "generic_webpage": "Página web Genérica",
    #                 "facebook": "Facebook (Próximamente)",
    #                 "instagram": "Instagram (Próximamente)",
    #                 "amazon_reviews": "Amazon Reviews (Próximamente)"
    #             }
    #         ),
            
    #         # Inputs dinámicos según plataforma seleccionada
    #         ui.output_ui("platform_inputs"),
            
    #         #ui.input_action_button("execute", "Ejecutar", class_="btn-primary"),
    #         ui.input_action_button("execute", " Scrapear!!", icon=ui.tags.i(class_="fas fa-play"), class_="btn-primary"),
    #         width=350
    #     ),
        
    #     ui.navset_card_tab(
    #         ui.nav_panel(
    #             " Base de datos",
    #             ui.output_data_frame('df_data'),
    #             #ui.download_button("download_data", "Descargar CSV", class_="btn-info btn-sm mb-2")
    #             ui.download_button("download_data", "Descargar CSV", icon=ui.tags.i(class_="fas fa-download"), class_="btn-info btn-sm mb-2 mt-2"),
    #             icon=ui.tags.i(class_="fas fa-table-list")                
    #             #ui.output_ui("dynamic_content")
    #             #ui.output_ui('platform_inputs')
    #         ),
    #         ui.nav_panel(
    #             " Resumen",
    #             #ui.output_text("summary_output"),
    #             ui.output_ui('styled_summary_output'),
    #             icon=ui.tags.i(class_="fas fa-file-lines")

    #         ),
    #         ui.nav_panel(
    #             " Análisis de Sentimiento",
    #             #output_widget("sentiment_output"),
    #             ui.output_plot("sentiment_output"),
    #             ui.download_button("download_sentiment_plot", "Descargar Gráfico (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"),                
    #             #ui.output_ui('styled_summary_output'),
    #             #icon=ui.tags.i(class_="fa-solid fa-chart-simple")
    #             #icon=ui.tags.i(class_="fa-solid fa-face-smile"),
    #             #icon=ui.tags.i(class_="fa-solid fa-face-frown")
    #             icon=ui.tags.i(class_="fa-solid fa-magnifying-glass-chart")
    #         ),
    #         ui.nav_panel(
    #             " Análisis de Emociones",
    #             ui.output_plot("emotion_plot_output"),
    #             ui.download_button("download_emotion_plot", "Descargar Gráfico de Emociones (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"),
    #             #icon=ui.tags.i(class_="fa-solid fa-face-grin-beam"), 
    #             #icon=ui.tags.i(class_="fa-solid fa-face-sad-cry"),
    #             #icon=ui.tags.i(class_="fa-solid fa-face-angry")
    #             icon=ui.tags.i(class_="fa-solid fa-icons")
    #         ),
    #         ui.nav_panel(
    #             "Map (Solo con Google Maps Selector)",
    #             ui.output_ui("google_map_embed"),
    #             icon = ui.tags.i(class_="fas fa-map-marked-alt")
    #         ),
    #         ui.nav_panel(
    #             "Análisis de tópicos",
    #             ui.output_plot("topics_plot_output"),
    #             ui.download_button("download_topics_plot", "Descargar Gráfico de Tópicos (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"),
    #             icon = ui.tags.i(class_="fa-solid fa-chart-bar")
    #         ),
    #         ui.nav_panel(
    #             " Chat con Gemini",
    #             ui.layout_sidebar(
    #                 ui.sidebar(
    #                     ui.input_text_area("gemini_prompt", "Tu pregunta:", 
    #                                      placeholder="Escribe tu pregunta para Gemini aquí..."),
    #                     #ui.input_action_button("ask_gemini", "Preguntar", class_="btn-success"),
    #                     ui.input_action_button("ask_gemini", "Preguntar", icon=ui.tags.i(class_="fas fa-paper-plane"),
    #                                             class_="btn-success"),
    #                     width=350
    #                 ),
    #                 ui.card(
    #                     ui.card_header("Respuesta de Gemini"),
    #                     ui.output_text("gemini_response"),
    #                     height="400px",
    #                     style="overflow-y: auto;"
    #                 )
    #             ),
    #             icon=ui.tags.i(class_="fa-solid fa-robot")
    #         ),
    #         ui.nav_panel(
    #             " Mapa Mental", 
    #             ui.output_ui("mind_map_output"),
    #             icon=ui.tags.i(class_="fas fa-project-diagram")
    #         ),   
    #         ui.nav_panel(
    #             #" Web Scraper/Parser",
    #             " Scrapear Tablas con chatbot",

    #             ui.layout_sidebar(
    #                 ui.sidebar(
    #                     ui.input_text_area("scraper_urls", "URLs a Scrapear (una sola): \n No admite redes sociales, solo páginas web", 
    #                                      placeholder="https://ejemplo.com/pagina", value ="https://www.elektra.mx/italika",  height=150),
    #                     ui.input_text_area("parser_description", "Describe qué información quieres extraer:", 
    #                                      placeholder="Ej: 'Tabla de precios de productos'", height=100, value = 'Genera una tabla con los precios de las motos de mayor a menor precio'),
    #                     ui.input_action_button("execute_scraper_parser", "Scrapear y Parsear", 
    #                                            icon=ui.tags.i(class_="fas fa-play"), class_="btn-primary"),
    #                     width=350
    #                 ),
    #                 ui.card(
    #                     ui.card_header("Resultados del Scraper y Parser"),
    #                     #ui.p("Para este caso, no se necesita seleccionar una plataforma del menú de la izquierda."),
    #                     # This output will dynamically show tables or raw text
    #                     ui.output_ui("scraper_parser_output"),
    #                     #height="600px", # Adjust height as needed
    #                     style="overflow-y: auto;"
    #                 )
    #             ),
    #             icon = ui.tags.i(class_="fa-solid fa-comments")
    #         )            
    #     )
    # ),
    #### """ui antiguo"""
    ui.output_ui("ui_app_dinamica"),
    #theme=shinyswatch.theme.darkly()
    theme=shinyswatch.theme.minty()
)

#### Comenzando el server/Backend
def server(input, output, session):
    # Variables reactivas para la autenticación
    usuario_autenticado = reactive.Value(None)
    mensaje_login = reactive.Value("")
    # Configuración inicial de las variables para el server
    ## Lazy Load o Carga del Perezoso
    pinecone_client = reactive.Value(None)
    pinecone_index_instance = reactive.Value(None)
    processed_dataframe = reactive.Value(pd.DataFrame())
    #current_gemini_response = reactive.Value("Escribe tu pregunta y presiona enviar")
    current_gemini_response = reactive.Value("Carga datos y luego haz una pregunta sobre ellos, o haz una pregunta general. Presiona Enter o el botón verde a la izquierda para activar el bot")
    gemini_embeddings_model = reactive.Value(None)
    gemini_model_instance= reactive.Value(None)
    spacy_nlp_sentiment = reactive.Value(None)
    pysentimiento_analyzer_instance = reactive.Value(None)
    pysentimiento_emotions_analyzer_instance = reactive.Value(None)    
    summarizer_pipeline_instance = reactive.Value(None)
    emotion_model = reactive.Value(None)
    emotion_tokenizer = reactive.Value(None)
    emotion_config = reactive.Value(None)
    scraper_parser_results = reactive.Value(None)
    llm_model_instance = reactive.Value(None)
    topic_pipeline_instance =  reactive.Value(None)
    map_coordinates = reactive.Value(None)
    mind_map_html = reactive.Value(None)
    
    # --- Comenzando la lógica para autenticación
    def ui_login_form():
        """Retorna la UI para el formulario de login."""
        return ui.div(
            ui.row(
                ui.column(4,
                          ui.panel_well(
                              ui.h3("Acceso Social Media Downloader", style="text-align: center;"),
                              ui.hr(),
                              ui.input_text("email_login", "Correo Electrónico:", placeholder="usuario@dominio.com"),
                              ui.input_action_button("boton_login", "Ingresar", class_="btn-primary btn-block"),
                              ui.output_text("texto_mensaje_login"),
                              style="color: #00968b; margin-top: 10px; text-align: center;"
                          ),
                        offset=4
                ) # Fin ui.column
            ),
            style="margin-top: 100px;"
        )

    @output
    @render.text
    def texto_mensaje_login():
        return mensaje_login.get()

    @reactive.Effect
    @reactive.event(input.boton_login)
    def manejar_intento_login():
        email = input.email_login()
        if not email:
            mensaje_login.set("Por favor, ingrese su correo electrónico.")
            return

        try:
            nombre_usuario, dominio = email.strip().lower().split('@')
            if dominio in DOMINIOS_PERMITIDOS:
                usuario_autenticado.set(nombre_usuario) # Guardamos solo la parte antes del @
                mensaje_login.set("")
                ui.notification_show(f"¡Bienvenido, {nombre_usuario}!", type="message", duration=5)
            else:
                mensaje_login.set("Dominio de correo no autorizado.")
                usuario_autenticado.set(None)
        except ValueError:
            mensaje_login.set("Formato de correo electrónico inválido.")
            usuario_autenticado.set(None)
        except Exception as e:
            mensaje_login.set(f"Error inesperado durante el login: {e}")
            usuario_autenticado.set(None)

    @reactive.Effect
    @reactive.event(input.boton_logout) # Necesitarás añadir este botón en tu UI principal
    def manejar_logout():
        nombre_usuario_actual = usuario_autenticado.get()
        usuario_autenticado.set(None)
        mensaje_login.set("Sesión cerrada exitosamente.")
        if nombre_usuario_actual:
            ui.notification_show(f"Hasta luego, {nombre_usuario_actual}.", type="message", duration=5)
        else:
            ui.notification_show("Sesión cerrada.", type="message", duration=5)


    # --- UI Principal de la Aplicación (tu UI original reestructurada) ---
    @reactive.Calc
    def ui_principal_app():
        """Retorna la UI principal de la aplicación cuando el usuario está autenticado."""
        return ui.layout_sidebar(
            ui.sidebar(
                ui.output_ui("sidebar_dinamico"), # Contenido del sidebar cambiará
                width=350
            ),
            ui.navset_card_tab( # Pestañas principales
                nav_panel_base_datos_y_chatbot(),
                nav_panel_analisis_y_visualizaciones(),
                nav_panel_scraper_tablas_chatbot(),
                id="pestana_principal_seleccionada"
            )
        )

    # --- Renderizador Condicional de UI ---
    @output
    @render.ui
    def ui_app_dinamica():
        if usuario_autenticado.get() is None:
            return ui_login_form()
        else:
            return ui_principal_app()

    # --- Sidebar Dinámico ---
    @output
    @render.ui
    def sidebar_dinamico():
        pestana_actual = input.pestana_principal_seleccionada()
        if pestana_actual == "Scrapear Tablas con Chatbot":
            return ui.div(
                ui.markdown(f"Usuario: **{usuario_autenticado.get()}**"),
                ui.markdown("**Scraper de Tablas con LLM**"),
                ui.hr(),
                ui.input_text_area("scraper_urls", "URLs a Scrapear (una sola): \n No admite redes sociales, solo páginas web", 
                                 placeholder="https://ejemplo.com/pagina", value ="https://www.elektra.mx/italika",  height=150),
                ui.input_text_area("parser_description", "Describe qué información quieres extraer:", 
                                 placeholder="Ej: 'Tabla de precios de productos'", height=100, value = 'Genera una tabla con los precios de las motos de mayor a menor precio'),
                ui.input_action_button("execute_scraper_parser", "Scrapear y Parsear", 
                                       icon=ui.tags.i(class_="fas fa-play"), class_="btn-primary"),
                ui.hr(),
                ui.input_action_button("boton_logout", "Cerrar Sesión", class_="btn-danger btn-sm btn-block")
            )
        else: # Para "Base de Datos y Chatbot" y "Análisis y Visualizaciones"
            return ui.div(
                ui.markdown(f"Usuario: **{usuario_autenticado.get()}**"),
                ui.markdown("**Social Media Downloader** - Extrae y analiza datos de diferentes plataformas."),
                ui.hr(),
                ui.input_select("platform_selector", "Seleccionar Plataforma:",
                    {"wikipedia": "Wikipedia", "playstore": "Google Play Store", "youtube": "YouTube", "maps": "Google Maps", "reddit": "Reddit", "twitter": "Twitter (X)", "generic_webpage": "Página web Genérica", "facebook": "Facebook (Próximamente)", "instagram": "Instagram (Próximamente)", "amazon_reviews": "Amazon Reviews (Próximamente)"}),
                ui.output_ui("platform_inputs"),
                ui.input_action_button("execute", " Scrapear!!", icon=ui.tags.i(class_="fas fa-play"), class_="btn-primary"),
                ui.hr(),
                ui.input_action_button("boton_logout", "Cerrar Sesión", class_="btn-danger btn-sm btn-block")
            )

    # --- Definiciones de las Pestañas Principales y Sub-Pestañas ---
    def nav_panel_base_datos_y_chatbot():
        return ui.nav_panel(
            "Base de Datos y Chatbot",
            ui.navset_card_pill( # Usamos pill para sub-pestañas
                ui.nav_panel("Base de Datos", 
                             ui.output_data_frame('df_data'),
                             ui.download_button("download_data", "Descargar CSV", icon=ui.tags.i(class_="fas fa-download"), class_="btn-info btn-sm mb-2 mt-2"),
                             icon=ui.tags.i(class_="fas fa-table-list")),
                ui.nav_panel("Resumen General", 
                             ui.output_ui('styled_summary_output'), 
                             icon=ui.tags.i(class_="fas fa-file-lines")),
                ui.nav_panel("Chat con Gemini", 
                             # icon=ui.tags.i(class_="fa-solid fa-robot"), # Movido después del layout
                             ui.layout_sidebar(
                                 ui.sidebar(
                                     ui.input_text_area("gemini_prompt", "Tu pregunta:", placeholder="Escribe tu pregunta para Gemini aquí..."),
                                     ui.input_action_button("ask_gemini", "Preguntar", icon=ui.tags.i(class_="fas fa-paper-plane"), class_="btn-success"),
                                     width=350 # Ancho del sidebar interno del chat
                                 ),
                                 ui.card(
                                     ui.card_header("Respuesta de Gemini"),
                                     ui.output_text("gemini_response"),
                                     height="400px", style="overflow-y: auto;"
                                 )
                             ), 
                             icon=ui.tags.i(class_="fa-solid fa-robot") # Icono al final
                ),
                ui.nav_panel("Mapa Geográfico (Solo al seleccionar Google Maps)", 
                             ui.output_ui("google_map_embed"), 
                             icon=ui.tags.i(class_="fas fa-map-marked-alt"))
            ),
            icon=ui.tags.i(class_="fas fa-database")
        )

    def nav_panel_analisis_y_visualizaciones():
        return ui.nav_panel(
            "Análisis y Visualizaciones",
            ui.navset_card_pill(
                ui.nav_panel("Análisis de Sentimiento", 
                             ui.output_plot("sentiment_output"), 
                             ui.download_button("download_sentiment_plot", "Descargar Gráfico (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"),
                             icon=ui.tags.i(class_="fa-solid fa-magnifying-glass-chart")),
                ui.nav_panel("Análisis de Emociones", 
                             ui.output_plot("emotion_plot_output"), 
                             ui.download_button("download_emotion_plot", "Descargar Gráfico de Emociones (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"),
                             icon=ui.tags.i(class_="fa-solid fa-icons")),
                ui.nav_panel("Análisis de Tópicos", 
                             ui.output_plot("topics_plot_output"), 
                             ui.download_button("download_topics_plot", "Descargar Gráfico de Tópicos (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"),
                             icon=ui.tags.i(class_="fa-solid fa-chart-bar")),
                ui.nav_panel("Mapa Mental", 
                             ui.output_ui("mind_map_output"), 
                             icon=ui.tags.i(class_="fas fa-project-diagram"))
            ),
            icon=ui.tags.i(class_="fas fa-chart-pie")
        )

    def nav_panel_scraper_tablas_chatbot():
        return ui.nav_panel(
            "Scrapear Tablas con Chatbot",
            # El contenido principal es posicional
            ui.card(
                ui.card_header("Resultados del Scraper y Parser"),
                ui.output_ui("scraper_parser_output"),
                style="overflow-y: auto;"
            ),
            icon=ui.tags.i(class_="fa-solid fa-wand-magic-sparkles") # El argumento de palabra clave 'icon' va al final
        )

    # --- Resto de tu lógica de servidor ---


    @render.image
    def app_logo():
        image_path = Path(__file__).parent / "www"/"LogoNuevo.png"
        return {"src": str(image_path), "alt": "App Logo"}
    #def _load_emotion_model():
    #    model_path = "daveni/twitter-xlm-roberta-emotion-es"
    #    try:
    #        tokenizer = AutoTokenizer.from_pretrained(model_path)
    #        config = AutoConfig.from_pretrained(model_path)
    #        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    #        emotion_tokenizer.set(tokenizer)
    #        emotion_config.set(config)
    #        emotion_model.set(model)     
    #        return True 
    #    except Exception as e:
    #        print(f"Error al cargar el modelo de emociones: {e}")
    #        return False
    
    def _ensure_spacy_sentiment_model():
        if spacy_nlp_sentiment.get() is None:
            try:
                print('Iniciando el modelo de Spacy')
                nlp = spacy.load('es_core_news_md')
                if not nlp.has_pipe('spacytextblob'):
                    nlp.add_pipe('spacytextblob')
                spacy_nlp_sentiment.set(nlp)
                print('Modelo Spacy cargado')
                return True
            except Exception as e:
                print(f"Error al cargar el modelo de Spacy")
                return False 
        return True
    
    def _ensure_pysentimiento_analyzer():
        if pysentimiento_analyzer_instance.get() is None:
            try:
                print('Iniciando el modelo pysentimiento para sentimientos (valga la redundancia)')
                analyzer = create_analyzer(task="sentiment", lang="es")
                pysentimiento_analyzer_instance.set(analyzer)
                print('Modelo pysentimiento para sentimientos cargado')
                return True
            except Exception as e:
                print(f"Error al cargar el modelo de pysentimiento para sentimientos {e}")
                return False
        return True

    def _ensure_pysentimient_emotions_analyzer():
        if pysentimiento_emotions_analyzer_instance.get() is None:
            try:
                print('Iniciando el modelo pysentimiento para emociones')
                analyzer = create_analyzer(task="emotion", lang="es")
                pysentimiento_emotions_analyzer_instance.set(analyzer)
                print('Modelo pysentimiento para emociones cargado')
                return True
            except Exception as e:
                print(f"Error al cargar el modelo de pysentimiento para emociones {e}")
                return False
        return True


    ## Para Gemini
    def _ensure_gemini_embeddings_model():
        if gemini_embeddings_model.get() is None:
            if GEMINI_API_KEY:
                try:
                    print("Iniciando el Gemini 'Embedding'")
                    genai.configure(api_key=GEMINI_API_KEY)
                    gemini_embeddings_model.set("models/embedding-001") 
                    print("Modelo Gemini 'Embedding' cargado")
                    return True
                except Exception as e:
                    print(f"Error al configurar Gemini {e}")
                    return False    
            else:
                return False 
        return True 
    
    ## Para Pinecone
    def _ensure_pinecone_client_and_index():
        if pinecone_index_instance.get() is None: 
            if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
                print('Error: Pinecone API o indice')
                return False
            try:
                print("Se inicia el cliente de Pinecone")
                #pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
                pc = Pinecone(api_key=PINECONE_API_KEY)

                pinecone_client.set(pc)
                if PINECONE_INDEX_NAME not in pc.list_indexes().names():
                    print(f"El indice pinecone {PINECONE_INDEX_NAME} no encontrado")
                    pc.create_index(
                        name = PINECONE_INDEX_NAME,
                        dimension = EMBEDDING_DIMENSION, 
                        metric= "cosine", 
                        spec = ServerlessSpec(
                            cloud ='aws',
                            region = 'us-east-1'
                        )
                    )
                    while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                        print("Waiting for Pinecone index to be ready...")
                        time.sleep(5)
                else: 
                    print('Se encontró el índice')
                index = pc.Index(PINECONE_INDEX_NAME)
                pinecone_index_instance.set(index)
                print(f"EL índice de Pinecone {PINECONE_INDEX_NAME} se ovbtuvo")
                return True 
            except Exception as e:
                print(f"Error al configurar Pinecone {e}")
                import traceback
                traceback.print_exc()
                return False
        return True 
    
    def embed_texts_gemini(texts: TypingList[str],  task_type="RETRIEVAL_DOCUMENT") -> TypingList[TypingList[float]]:
        if not _ensure_gemini_embeddings_model():
            raise Exception('Los embeddings de Gemini no están disponibles')
        
        model_name = gemini_embeddings_model.get()
        try:
            result = genai.embed_content(model=model_name, content=texts, task_type=task_type)
            return result['embedding']
        except Exception as e:
            print('Error al generar los embeddings de Gemini')
            return [[] for _ in texts]






    def generate_zero_shot_classification(text):
        candidate_labels_list = calculate_topic_text()
        if not _ensure_topics_pipeline():
            print("Error: Modelo de clasificación de temas no disponible")
            return "Error interno modelo de tópicos"
        classifier = topic_pipeline_instance.get()
        
        text = str(text) if pd.notna(text) else ""

        if not text: 
            return "Texto vacío o inválido" 

        if classifier is None:
            print("Error: Modelo de clasificación de temas no cargado")
            return "Error: Pipeline de clasificación de temas no disponible"

        # Ensure candidate_labels is a list
        if not isinstance(candidate_labels_list, list):
            print(f"Warning: candidate_labels is not a list: {candidate_labels_list}. Attempting conversion.")
            try:
                actual_labels = ast.literal_eval(str(candidate_labels_list))
                if not isinstance(actual_labels, list):
                    raise ValueError("Conversion did not result in a list")
            except Exception as e:
                print(f"Error processing candidate_labels: {e}")
                return "Error interno modelo de clasificación"

        try:
            result = classifier(text, candidate_labels_list)
            # The result structure is a dictionary, result['labels'] is a list of labels sorted by score
            return result['labels'][0] # Return the top label
        except Exception as e:
            print(f"Error processing text because empty row '{text[:50]}...': {e}. Returning 'No Aplica'.")
            return "No Aplica" 


    ### Insertamos las funciones generales que se usarán en toda la app 
    def generate_sentiment_analysis(text):
        text = str(text)
        #if not _ensure_spacy_sentiment_model():
        #    return "Error: Modelo de sentimiento no disponible"
        #nlp_model = spacy_nlp_sentiment.get()
        #if nlp_model is None: 
        #    return "Error interno modelo de sentimiento"
        ##doc = nlp_sentiment(text)
        #doc = nlp_model(text)
        #polarity = doc._.blob.polarity  
        #sentiment='Neutral'
        #if polarity > 0.1:
        #    sentiment = 'Positivo'
        #elif polarity < -0.1:
        #    sentiment = 'Negativo'
        #return sentiment
        use_spacy = _ensure_spacy_sentiment_model()
        nlp_model = spacy_nlp_sentiment.get() if use_spacy else None

        if nlp_model:
            doc = nlp_model(text)
            polarity_spacy = doc._.blob.polarity

            if polarity_spacy > 0.1:
                return 'Positivo'  
            elif polarity_spacy < -0.1:
                return 'Negativo' 
            else:
                # Si Spacy es neutral, usa otro método
                sentiment_spacy_neutral = 'Neutral'
                print(f"SpacyTextBlob polarity ({polarity_spacy:.3f}) is neutral/weak. Consulting Pysentimiento for: '{text[:60]}...'")

                use_pysentimiento = _ensure_pysentimiento_analyzer()
                analyzer_pysent = pysentimiento_analyzer_instance.get() if use_pysentimiento else None

                if analyzer_pysent:
                    try:
                        result_pysentimiento = analyzer_pysent.predict(text)
                        pysent_sentiment_map = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutral"}
                        sentiment_pysent = pysent_sentiment_map.get(result_pysentimiento.output, "Neutral")

                        if sentiment_pysent != "Neutral":
                            print(f"Pysentimiento valores: {sentiment_pysent} (Probas: {result_pysentimiento.probas})")
                            return sentiment_pysent
                        else:
                            print(f"Pysentimiento también dice que es neutral: {result_pysentimiento.probas}")
                            return sentiment_spacy_neutral
                    except Exception as e:
                        print(f"Error en pysentimiento: {e}. Tomando el valor de Spacy ({sentiment_spacy_neutral}).")
                        return sentiment_spacy_neutral
                else: 
                    print("Pysentimiento no disponible. Se queda valor neutro")
                    return sentiment_spacy_neutral
        else:
            print("Spacy falló, usar Pysentimiento.")
            use_pysentimiento = _ensure_pysentimiento_analyzer()
            analyzer_pysent = pysentimiento_analyzer_instance.get() if use_pysentimiento else None

            if analyzer_pysent:
                try:
                    result = analyzer_pysent.predict(text)
                    pysent_sentiment_map = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutral"}
                    return pysent_sentiment_map.get(result.output, "Neutral")
                except Exception as e:
                    print(f"Error en pysentimiento: {e}")
                    return "Error: Análisis de sentimiento (Pysentimiento) fallido"
            else: # Both Spacy and Pysentimiento are unavailable
                return "Error: Modelos de sentimiento no disponibles"        


    def generate_emotions_analysis(text):
        text = str(text)
        if not _ensure_pysentimient_emotions_analyzer():
            return "Error: Modelo de emociones no disponible"
        analyzer = pysentimiento_emotions_analyzer_instance.get()
        if analyzer is None:
            return "Error interno modelo de emociones"
        emotion_map_es = {
            "joy": "Alegría", 
            "sadness": "Tristeza", 
            "anger": "Enojo", 
            "fear": "Miedo", 
            "surprise": "Sorpresa", 
            "disgust": "Asco", 
            "neutral": "Neutral"

        }    
        try:
            result = analyzer.predict(text)
            primary_emotion_en = result.output
            if primary_emotion_en=="others":
                primary_emotion_en="neutral"
            return emotion_map_es.get(primary_emotion_en, "Desconocida")
        except Exception as e:
            print(f"Error en el análisis de emociones {e}")
            return "Error: Análisis de emociones fallido"

    #@reactive.calc
    #def detectEmotion(text):
    #    ### Obtiene el modelo preentrenado
    #    #model_path = "daveni/twitter-xlm-roberta-emotion-es"
    #    #tokenizer = AutoTokenizer.from_pretrained(model_path )
    #    #config = AutoConfig.from_pretrained(model_path )
    #    #emotions_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    #    model = emotion_model.get()
    #    tokenizer = emotion_tokenizer.get()
    #    config = emotion_config.get()
    #    if model is None or tokenizer is None or config is None:
    #        return "Modelo no cargado", None
    #    ### Starting the encoding
    #    text = str(text)
    #    encoded_input = tokenizer(text, return_tensors='pt')
    #    try:
    #        #output = emotions_model(**encoded_input)
    #        output = model(**encoded_input)
    #        scores = output[0][0].detach().numpy()
    #        scores = softmax(scores)
    #        ranking = np.argsort(scores)
    #        ranking = ranking[::-1]
    #        emotions_score = np.sort(range(scores.shape[0]))
    #        emotions_score= emotions_score[0]
    #        l = config.id2label[ranking[emotions_score]]
    #        s = scores[ranking[emotions_score]]
    #        if l=='others':
    #            l='neutral'
    #        return l, np.round(float(s), 4)
    #    except:
    #        return None, None


    def collapse_text(df):
        try: 
            if 'text' in df.columns: 
                total_text = df['text'].astype(str)
                joined_text = " ".join(total_text.dropna())
                return joined_text
            elif 'comment' in df.columns:
                #total_text = df['comment']
                #joined_text = " ".join(total_text)
                total_text = df['comment'].astype(str)
                joined_text = " ".join(total_text.dropna())
                return joined_text
            elif 'content' in df.columns:
                total_text = df['content'].astype(str)
                joined_text = " ".join(total_text.dropna())
                return joined_text
            else:
                #return None
                return ""   
        except Exception as e:
            return f"Error al intentar unir el texto: {e}"          


    def scrape_website(website_url):
        print(f"Attempting to scrape: {website_url}")
        
        unique_user_data_dir = None  # Initialize
        driver = None               # Initialize

        # Select a random User-Agent for this attempt
        selected_user_agent = random.choice(user_agents)        
        try:
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--ignore-certificate-errors")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-infobars")
            options.add_argument(f"user-agent={selected_user_agent}") # Set the random User-Agent
            unique_user_data_dir = tempfile.mkdtemp() # Create dir
            options.add_argument(f"--user-data-dir={unique_user_data_dir}")

            driver_executable_path = None
            prepare_driver_error = None

            # 1. Determine ChromeDriver path and ensure permissions
            if not CHROME_DRIVER_PATH:
                print("CHROME_DRIVER_PATH not set. Using webdriver-manager.")
                try:
                    # Ensure webdriver_manager is imported if this path is taken
                    from webdriver_manager.chrome import ChromeDriverManager
                    driver_executable_path = ChromeDriverManager().install()
                    abs_driver_path = os.path.abspath(driver_executable_path)
                    print(f"webdriver-manager installed/found chromedriver at: {abs_driver_path}")
                    if os.name == 'posix' and os.path.exists(abs_driver_path):
                        print(f"Setting execute permissions (0o755) for: {abs_driver_path}")
                        os.chmod(abs_driver_path, 0o755)
                        if not os.access(abs_driver_path, os.X_OK):
                            prepare_driver_error = f"Failed to make webdriver-manager's chromedriver executable at {abs_driver_path} even after chmod. Check for 'noexec' mount."
                            print(prepare_driver_error)
                        else:
                            print(f"Successfully set execute permissions for webdriver-manager driver: {abs_driver_path}")
                except Exception as e_wdm:
                    prepare_driver_error = f"ChromeDriverManager failed: {str(e_wdm)}"
                    print(prepare_driver_error)
            else:
                initial_cd_path_env = CHROME_DRIVER_PATH
                
                # Resolve CHROME_DRIVER_PATH. os.path.abspath resolves based on CWD.
                # If CHROME_DRIVER_PATH is just "chromedriver", and script is run from its dir, this works.
                abs_driver_path = os.path.abspath(initial_cd_path_env)
                
                print(f"--- ChromeDriver Path Resolution (CHROME_DRIVER_PATH is set) ---")
                print(f"CHROME_DRIVER_PATH (initial from env): '{initial_cd_path_env}'")
                print(f"Current Working Directory (os.getcwd()): '{os.getcwd()}'")
                print(f"Script Directory (Path(__file__).parent): '{Path(__file__).parent}'")
                print(f"Resolved Absolute Path for chromedriver: '{abs_driver_path}'")
                print(f"--- End ChromeDriver Path Resolution ---")
                
                driver_executable_path = abs_driver_path # Use the absolute path from now on

                if not os.path.exists(driver_executable_path):
                    prepare_driver_error = f"CHROME_DRIVER_PATH specified but resolved path not found: {driver_executable_path}"
                    print(prepare_driver_error)
                elif not os.path.isfile(driver_executable_path):
                    prepare_driver_error = f"CHROME_DRIVER_PATH specified but resolved path is not a file: {driver_executable_path}"
                    print(prepare_driver_error)
                elif os.name == 'posix':
                    try:
                        print(f"Checking execute permissions for user-provided: {driver_executable_path}")
                        if not os.access(driver_executable_path, os.X_OK):
                            current_mode = os.stat(driver_executable_path).st_mode
                            print(f"File {driver_executable_path} is not executable. Current mode: {oct(current_mode)}. Attempting to add execute permission.")
                            new_mode = current_mode | os.S_IXUSR | os.S_IXGRP | os.S_IXOTH
                            os.chmod(driver_executable_path, new_mode)
                            if os.access(driver_executable_path, os.X_OK):
                                print(f"Successfully set execute permission for {driver_executable_path}. New mode: {oct(os.stat(driver_executable_path).st_mode)}")
                            else:
                                prepare_driver_error = (f"Failed to set execute permission for {driver_executable_path} despite attempt. "
                                                        "Check script's permissions to chmod this file and ensure filesystem is not mounted with 'noexec'.")
                                print(prepare_driver_error)
                        else:
                            print(f"File {driver_executable_path} is already executable. Mode: {oct(os.stat(driver_executable_path).st_mode)}")
                    except Exception as e_chmod:
                        prepare_driver_error = f"Error during permission check/set for {driver_executable_path}: {e_chmod}. Ensure the script has rights to chmod this file."
                        print(prepare_driver_error)

            # 2. Attempt Selenium scraping if driver is ready
            selenium_content = None
            selenium_attempt_error = prepare_driver_error # Carry over error from driver prep

            if not selenium_attempt_error and driver_executable_path: # If driver path is valid and no prep error
                try:
                    print("Initializing Selenium WebDriver...")
                    service = Service(executable_path=driver_executable_path)
                    driver = webdriver.Chrome(service=service, options=options)
                    
                    print(f"Navigating to {website_url} with Selenium...")
                    driver.get(website_url) # This can timeout
                    print("Page loaded with Selenium.")
                    selenium_content = driver.page_source
                except TimeoutException as e_timeout:
                    selenium_attempt_error = f"Selenium navigation timed out for {website_url}: {str(e_timeout)}"
                    print(selenium_attempt_error)
                except Exception as e_selenium: # Catches other Selenium errors, including permission issues if chmod failed silently before
                    selenium_attempt_error = f"Selenium WebDriver error (executable: {driver_executable_path}): {str(e_selenium)}"
                    print(selenium_attempt_error)
            elif not selenium_attempt_error and not driver_executable_path: # Should not happen if logic is correct
                 selenium_attempt_error = "ChromeDriver path could not be determined despite no direct error during preparation."
                 print(selenium_attempt_error)

            # 3. Evaluate Selenium content or error
            if selenium_content:
                access_denied_keywords = [
                    "You don't have permission to access", "Reference #18.",
                    "403 Forbidden", "Access Denied"
                ]
                if any(keyword.lower() in selenium_content.lower() for keyword in access_denied_keywords):
                    print("Access denied content detected in Selenium page source. Will attempt fallback.")
                    selenium_attempt_error = selenium_attempt_error or "Access denied via Selenium."
                else:
                    print("Selenium scraping successful.")
                    return selenium_content # SUCCESSFUL SELENIUM EXIT
            
            # 4. Fallback to requests if Selenium failed, was skipped, or yielded denied content
            print(f"Attempting fallback with requests for {website_url} (Reason: {selenium_attempt_error or 'Selenium did not yield usable content'})...")
            try:
                agent = {'User-Agent': selected_user_agent}
                page = requests.get(website_url, headers=agent, timeout=15)
                page.raise_for_status()
                print("Requests fallback successful.")
                return page.content # SUCCESSFUL REQUESTS EXIT
            except requests.exceptions.RequestException as req_e:
                requests_error_message = f"Requests fallback failed: {str(req_e)}"
                print(requests_error_message)
                final_error_message = f"Error scraping {website_url}. "
                if selenium_attempt_error:
                    final_error_message += f"Selenium Error: {selenium_attempt_error}. "
                final_error_message += f"Requests Error: {requests_error_message}."
                return final_error_message

        finally:
            if driver:
                print("Quitting Selenium driver...")
                driver.quit()
            if unique_user_data_dir:
                import shutil
                print(f"Cleaning up temp dir: {unique_user_data_dir}")
                try:
                    shutil.rmtree(unique_user_data_dir)
                    print(f"Successfully cleaned up temp dir: {unique_user_data_dir}")
                except OSError as e_shutil:
                    print(f"Error removing temp dir {unique_user_data_dir}: {e_shutil}")


    def extract_body_content(html_content):
        if not html_content: # Handles empty string or None
            return ""
        # Check if html_content is an error string from scrape_website
        if isinstance(html_content, str) and html_content.startswith("Error scraping"):
            return html_content # Pass the error string through
        
        try:
            # BeautifulSoup can handle bytes (it will try to auto-detect encoding) or a string.
            soup = BeautifulSoup(html_content, "html.parser")
            body_content = soup.body
            if body_content:
                return str(body_content)
            return "" # No body tag found
        except Exception as e:
            print(f"Error in extract_body_content: {e}")
            return f"Error extracting body: {e}"

    def clean_body_content(body_content):
        if not body_content:
            return ""
        # Check if body_content is an error message from previous steps
        if (isinstance(body_content, str) and 
            (body_content.startswith("Error scraping") or body_content.startswith("Error extracting body"))):
            return body_content # Pass the error string through

        try:
            soup = BeautifulSoup(body_content, "html.parser")

            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()
            
            cleaned_content = soup.get_text(separator="\n")
            cleaned_content = "\n".join(
                line.strip() for line in cleaned_content.splitlines() if line.strip()
            )
            print('Cleaning body')
            return cleaned_content
        except Exception as e:
            print(f"Error in clean_body_content: {e}")
            return f"Error cleaning body: {e}"

    def split_dom_content(dom_content,max_length=60000):
        # Pass through error messages
        if isinstance(dom_content, str) and (dom_content.startswith("Error scraping") or \
                                             dom_content.startswith("Error extracting body") or \
                                             dom_content.startswith("Error cleaning body")):
            return [dom_content]
        if not dom_content:
            return []
        return [
            dom_content[i:i+max_length] for i in range(0,len(dom_content),max_length)
        ]


    def _ensure_summarizer_pipeline():
        if summarizer_pipeline_instance.get() is None:
            try:
                print('Cargando pipeline de resumen')
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                summarizer_pipeline_instance.set(summarizer)
                print('Pipeline de resumen cargado')
                return True
            except Exception as e : 
                print(f"Error al cargar el pipeline de resumen {e}")
                return False 
        return True 
    
    def _ensure_topics_pipeline():
        if topic_pipeline_instance.get() is None: 
            try: 
                print('Cargando pipeline de tópicos')
                topicGenerator = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
                      #model="microsoft/deberta-v3-small")
                      #model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli") # Modelo multilenguaje
                topic_pipeline_instance.set(topicGenerator)
                print('Pipeline de tópicos cargado')
                return True             
            except Exception as e:
                print(f"Error al cargar el pipeline de tópicos {e}")
                topic_pipeline_instance.set(None) 
                return False 
        return True 
    

    def generate_zero_shot_classification_with_labels(text, candidate_labels):
        classifier = topic_pipeline_instance.get()
        text_to_classify = str(text) if pd.notna(text) else ""
        if not text_to_classify:
            return "Texto vacío o inválido"
        if classifier is None:  
            print('Error Topic Classification Pipeline no cargado')
            return "Error: Pipeline de clasificación de temas no disponible"
        if not candidate_labels or not isinstance(candidate_labels, list) or not all(isinstance(label, str) for label in candidate_labels):
            print(f"Error: candidate_labels is not a valid list of strings: {candidate_labels}")
            return "Error: Lista de etiquetas de clasificación no válida"               

        try: 
            result = classifier(text_to_classify, candidate_labels)
            return result['labels'][0]  
        except Exception as e:
            print(f'Errir al clasificar el texto: {e}')
            return 'No aplica'

    def summary_generator(text, platform):   
        #print('Entrando a la función de resumen')
        #summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        #if not _ensure_summarizer_pipeline():
        #    return "Error: Pipelin de resumen no disponible"
        #summarizer = summarizer_pipeline_instance.get()
        if not text:
            return "No hay texto para resumir"
        text = str(text)
        #if platform=="wikipedia" or platform=="generic_webpage":
        if platform=="wikipedia2":            
            if not _ensure_summarizer_pipeline():
                return "Error: Pipelin de resumen no disponible"
            summarizer = summarizer_pipeline_instance.get()
        #if text:
            try:
                #text = str(text)
                #print('Comenzando a resumir')
                #summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
                #print('Terminó de resumir')
                #return f"Resumen: \n{summary}"
                max_bart_input_len = 1024*3
                if len(text) > max_bart_input_len:
                    text_to_process = text[:max_bart_input_len]
                    print(f"Texto para resumen (BART) truncado a {max_bart_input_len} caracteres.")
                else: 
                    text_to_process = text
                print('Comenzando a resumir con BART')
                summary = summarizer(text_to_process, max_length=200, min_length=40, do_sample=False)[0]['summary_text']
                print('Resumen con BART terminado.')
                return f"Resumen (BART):\n{summary}"
            except Exception as e: 
                return f"Error al resumir con BART: {e}"
        else:
            #return "No hay texto para resumir."
            if not _ensure_gemini_model():
                return "Error: Modelo de Gemini no disponible"
            gemini_model = gemini_model_instance.get()
            max_gemini_input_len = 1000
            if len(text) > max_gemini_input_len:
                text_to_process = text[:max_gemini_input_len]
                print(f"Texto para resumen (Gemini) truncado a {max_gemini_input_len} caracteres.")
            else: 
                text_to_process= text
            summarization_prompt = (
                "Por favor, resume el siguiente texto extraído de una plataforma de red social. "
                "Concéntrate en las ideas principales y el sentimiento general si es evidente. "
                f"El texto es:\n\n---\n{text_to_process}\n---\n\nResumen conciso:"
            )
            try:
                print(f"Enviando a Gemini para resumen: {summarization_prompt[:200]}...")
                response = gemini_model.generate_content(summarization_prompt)
                return f"Resumen (Gemini):\n{response.text}"
            except Exception as e:
                return f"Error al resumir con Gemini: {e}"

    #### Añadido el 26 de Abril 2025
    def topics_generator(text):
        if not text:
            return "No hay texto para resumir"
        text = str(text)
        if not _ensure_gemini_model():
            return "Error: Modelo de Gemini no disponible"
        gemini_model = gemini_model_instance.get()
        max_gemini_input_len = 7000
        if len(text) > max_gemini_input_len:
            text_to_process = text[:max_gemini_input_len]
            print(f"Texto para resumen (Gemini) truncado a {max_gemini_input_len} caracteres.")
        else: 
            text_to_process= text
        topics_prompt = (f'''
            Analiza el siguiente texto:
            "{text}"
            Tu tarea es extraer:
            1. Tópicos generales que engloben a todo el texto
            2. Genera 5 categorías relevantes y sencillas (máximo 4 palabras cada una) que resuman los temas principales o aspectos del texto
            IMPORTANTE: Formatea TODA tu respuesta EXCLUSIVAMENTE como una ÚNICA cadena de texto que represente una lista de Python.
            Esta lista debe contener strings. Cada string puede ser una funcionalidad o una categoría.
            Ejemplo de formato de respuesta esperado: ['funcionalidad A', 'funcionalidad B', 'categoría X', 'categoría Y', ..., 'categoría Z']
            No incluyas NINGÚN texto, explicación, ni markdown (como ```python ... ```) antes o después de esta lista. Solo la lista en formato de cadena.
            ''' 
        )
        try:
            print(f"Enviando a Gemini para topicos: {topics_prompt[:20]}...")
            response = gemini_model.generate_content(topics_prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error al generar tópicos con Gemini: {e}"


    

    def extract_wikipedia_paragraphs(url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = [p.get_text() for p in soup.find_all('p')]
            return paragraphs
        except requests.exceptions.RequestException as e:
            return [f"Error al acceder a Wikipedia: {e}"]


    def generate_trigrams(text):
        nlp = spacy.load("es_core_news_md")
        doc = nlp(text.lower())
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
        trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
        return trigrams
    

    def detectLanguage(text, api_key):
        language_detected = single_detection(text, api_key=api_key)
        return language_detected


    def TranslateText(text, source, target):
        translatedText = GoogleTranslator(source = source, target = target).translate(text)
        return translatedText    
    

    @reactive.Effect
    @reactive.event(input.execute)
    def handle_execute():
        platform = input.platform_selector()
        df = pd.DataFrame()
        map_coordinates.set(None)
        with ui.Progress(min=1, max=10) as p:
            p.set(message="Procesando...", detail=f"Extrayendo datos de {platform}")
            if platform == "wikipedia":
                df = process_wikipedia_for_df()
            elif platform=="youtube":
                df = get_youtube_comments()
            elif platform=="maps":
                df = mapsComments()
            elif platform=="twitter":
                df = getTweetsResponses()
            elif platform=="generic_webpage":
                df = process_generic_webpage_for_df()
            elif platform=="reddit":
                df = get_reddit_comments()
            elif platform=="playstore":
                df = get_playstore_comments()
        processed_dataframe.set(df)

    @reactive.Effect
    @reactive.event(input.execute_scraper_parser)
    def handle_scraper_parser_execute():
        urls_input = input.scraper_urls()
        parse_description = input.parser_description()

        if not urls_input:
            scraper_parser_results.set({"error": "Por favor, ingresa al menos una URL."})
            return
        urls_list = [url.strip() for url in urls_input.splitlines() if url.strip()]
        if not urls_list:
            scraper_parser_results.set({"error": "Por favor, ingresa al menos una URL válida."})
            return
        if not parse_description:
            scraper_parser_results.set({"error": "Por favor, ingresa una descripción para el parseo."})
            return

        with ui.Progress(min=1, max=len(urls_list) + 3) as p:
            p.set(message="Procesando Scraper/Parser...", detail="Iniciando...")
            all_cleaned_content = []
            for i, url_item in enumerate(urls_list):
                p.set(i + 1, detail=f"Scraping {url_item}...")
                html = scrape_website(url_item)
                body = extract_body_content(html)
                cleaned = clean_body_content(body)
                all_cleaned_content.append(cleaned)

            p.set(len(urls_list) + 1, detail="Parseando contenido con LLM...")
            raw_llm_outputs = parse_content_with_llm(all_cleaned_content, parse_description)
            
            p.set(len(urls_list) + 2, detail="Extrayendo tablas...")
            all_extracted_tables = extract_tables_from_llm_outputs(raw_llm_outputs)

            p.set(len(urls_list) + 3, detail="Fusionando tablas...")
            merged_table_df = merge_extracted_tables_llm(all_extracted_tables, parse_description)
            
            scraper_parser_results.set({"raw_outputs": raw_llm_outputs, "extracted_tables": all_extracted_tables, "merged_table": merged_table_df})
            p.set(len(urls_list) + 3, detail="Completado.")
            

    ## Pinecone structure
    @reactive.Effect
    def _upsert_data_to_pinecone_on_df_change():
        df = processed_dataframe.get()
        platform = input.platform_selector()
        if isinstance(df, pd.DataFrame) and not df.empty:
            if 'Error' in df.columns and len(df)==1:
                print('Esto no se insertará en Pinecone porque hay un error')
                return 
            if 'Mensaje' in df.columns and len(df)==1:
                print('Esto no se insertará en Pinecone porque hay un warning')
                return 
            
            if not _ensure_pinecone_client_and_index() or not _ensure_gemini_embeddings_model():
                print("No se pudo insertar Pinecone: Cliente, indice, o embeddings no está listo")
                return
            text_column = None
        
            if 'text' in df.columns:
                text_column = 'text'
            elif 'comment' in df.columns: 
                text_column = 'comment'
            elif 'content' in df.columns: 
                text_column = 'content'

            if text_column:
                texts_to_embed = df[text_column].dropna().astype(str).tolist()
                if not texts_to_embed:
                    print("No hay texto válido a ingerir a Pinecone")
                    return

                print(f"Añadiendo {len(texts_to_embed)} textos a Pinecone de la plataforma: {platform}...")
                try:
                    embeddings = embed_texts_gemini(texts_to_embed)
                    vectors_to_upsert = []
                    for i, text in enumerate(texts_to_embed):
                        vector_id = f"{platform}_{df.index[i]}_{hash(text)}" 
                        vectors_to_upsert.append((vector_id, embeddings[i], {"text": text, "source_platform": platform}))
                    pinecone_index_instance.get().upsert(vectors=vectors_to_upsert)
                    print(f"Se insertó exitosamente {len(vectors_to_upsert)} vectors a Pinecone.")
                    ui.notification_show(f"Indexado {len(vectors_to_upsert)} los items a Pinecone.", duration=5, type="info")
                except Exception as e:
                    print(f"Error al insertar a Pinecone: {e}")
                    ui.notification_show(f"Error al indexar la base de datos  {e}", duration=5, type="error")            



    # Dynamic content display
    @output
    @render.ui
    def platform_inputs():
        platform = input.platform_selector()
        if platform == "wikipedia":
            return ui.input_text(
                "wikipedia_url", 
                "URL de Wikipedia:", 
                placeholder="https://es.wikipedia.org/wiki/Tema",
                value="https://es.wikipedia.org/wiki/Agapornis"
            )
        elif platform == "youtube":
            return ui.input_text(
                "youtube_url", 
                "URL de YouTube:", 
                placeholder="https://www.youtube.com/watch?v=ID",
                value = "https://www.youtube.com/watch?v=NJwZ7j5qB3Y"
            )
        elif platform == "maps":
            return ui.input_text(
                "maps_query", 
                "Buscar en Google Maps:", 
                placeholder="Url del sitio o nombre del lugar",
                value = "Museo Nacional de Antropología, Ciudad de México"
            )
        elif platform == "twitter":
            return ui.input_text(
                #"twitter_url", 
                #"URL de Twitter:", 
                #placeholder="https://twitter.com/usuario/status/ID",
                #value= '1914271651962700018'
                "twitter_query",
                "X antes Twitter (Usuario @, Hashtag #, o URL/ID de Tweet):",
                placeholder="@usuario, #hashtag, https://x.com/usuario/status/ID, o ID_numerico",
                value = "#china"
            )
        elif platform=="generic_webpage":
            return ui.input_text(
                "generic_webpage_url",
                "URL de la página web",
                placeholder="https:paginaweb.com",
                value ="https://www.movil.gs/iris-startup-lab" 
            )
        elif platform=='reddit':
            return ui.input_text(
                "reddit_url",
                "URL de Reddit", 
                placeholder="https://www.reddit.com/r/tema/comments/id/nombre_del_reddit/", 
                value = "https://www.reddit.com/r/cockatiel/comments/1kkuzpg/change_colour/"
            )
        elif platform=='playstore':
            return ui.input_text(
                "playstore_url",
                "URL de Play Store", 
                placeholder="com.nianticlabs.pokemongo", 
                value = "https://play.google.com/store/apps/details?id=mx.com.bancoazteca.bazdigitalmovil"
            )


    @output
    @render.data_frame
    def df_data():
        #platform = input.platform_selector()
        #if platform == "wikipedia":
        #    df= process_wikipedia_for_df()
        #elif platform == "youtube":
        #    df= get_youtube_comments()
        #elif platform == "maps":
        #    df= mapsComments()
        #elif platform =='twitter':
        #    df= getTweetsResponses()
        #else: 
        #    df= pd.DataFrame()
        #if isinstance(df, pd.DataFrame):
        #    return render.DataGrid(df, height= 350)
        #else: 
        #    return render.DataGrid(pd.DataFrame({"Error": ["Datos no disponibles"]}), height=350)
        df = processed_dataframe.get()
        if isinstance(df, pd.DataFrame) and not df.empty:
            if 'Error' in df.columns and len(df) == 1:
                return render.DataGrid(df, height=350)
            else:
                return render.DataGrid(df, height=350)
        elif isinstance(df, pd.DataFrame) and 'Error' in  df.columns:
            return render.DataGrid(df, height=350)
        elif input.platform_selector()== "scraper_parser_tab_internal":
            return render.DataGrid(pd.DataFrame({"Mensaje": ["Resultados en la pestaña 'Web Scraper/Parser'."]}), height=350)
        else: 
            return render.DataGrid(pd.DataFrame({"Mensaje": ["Seleccione una plataforma, ingrese los parámetros y presione 'Scrapear!!' para cargar datos."]}), height=350)

    @reactive.calc
    def process_wikipedia_for_df():
        if input.platform_selector() != "wikipedia":
            return pd.DataFrame()

        url = input.wikipedia_url()      
        if not url:
            return pd.DataFrame({'Error': ["Por favor ingresa una URL de Wikipedia"]})

        paragraphs = extract_wikipedia_paragraphs(url)
        if not paragraphs or (len(paragraphs)==1 and not  paragraphs[0].strip()):
            return pd.DataFrame({"Error": ["No se pudo encontrar texto en la URL proporcionada"]})

        data = {'paragraph_number': range(1, len(paragraphs) + 1),
                'text': paragraphs,
                'length': [len(p) for p in paragraphs]#,
                #'trigrams': [generate_trigrams(p) for p in paragraphs],
                #'sentiment': [generate_sentiment_analysis(p) for p in paragraphs],
                #'emotion': [generate_emotions_analysis(p) for p in paragraphs]
                }
        df = pd.DataFrame(data) 
        if 'text' not in df.columns:
            return pd.DataFrame({'Error': ["Error interno: Columna 'text' no encontrada en DataFrame de Wikipedia."]})
        df['text'] = df['text'].astype(str)
        is_error_message = df['text'].str.startswith("Error al acceder a Wikipedia:")
        is_empty_or_whitespace = df['text'].str.strip() == ""
        df = df[~(is_error_message | is_empty_or_whitespace)]

        if df.empty: 
            return pd.DataFrame({"Error": ["No se encontraron párrafos válidos en la URL proporcionada"]})
        #testText = str(df['text'].iloc[0])
        testText = df['text'].iloc[0]
        #print(f"Test Text para WIkipedia: {testText}")
        collapsed_wiki_text = collapse_text(df)
        actual_candidate_labels = ['General']
        if collapsed_wiki_text:
            if not _ensure_topics_pipeline():
                print("Error: Pipeline de tópicos no disponible")
            candidate_labels_str= topics_generator(collapsed_wiki_text)

            if isinstance(candidate_labels_str, str) and not candidate_labels_str.startswith("Error:"):
                try:
                    parsed_labels = ast.literal_eval(candidate_labels_str)
                    if isinstance(parsed_labels, list):
                        actual_candidate_labels = parsed_labels
                    else:   
                        print(f"Error: La respuesta de Gemini no es una lista válida: {candidate_labels_str}")  
                except (ValueError, SyntaxError) as e:
                    print(f"Error al procesar las etiquetas de tópicos: {e}")
            elif isinstance(candidate_labels_str, str) and candidate_labels_str.startswith("Error:"):
                print(f"Error: La respuesta de Gemini no es una lista válida: {candidate_labels_str}")

        def detectLanguageWiki(url):
            pattern = r"https?://([a-z]{2})\.wikipedia\.org"
            match = re.search(pattern, url)
            if match:
                return match.group(1)
            return None
        languageDetected = detectLanguageWiki(url)
        print(f"Lenguaje detectado para Wikipedia en URL: {languageDetected}")
        if not languageDetected:
            try: 
                languageDetected = detectLanguage(testText, api_key=DETECT_LANGUAGE_API_KEY)
                print(f"Lenguaje detectado para Wikipedia: {languageDetected}")
            except Exception as e:
                print(f"Error al detectar el lenguaje: {e}")
                languageDetected='es'

        if languageDetected=='es':
            df['sentiment'] = df['text'].apply(generate_sentiment_analysis)
            df['emotion'] = df['text'].apply(generate_emotions_analysis)
            classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
            df['topics'] = df['text'].apply(classify_call)
        else: 
            df['text'] = df['text'].apply(lambda x: TranslateText(text=x, source= languageDetected, target='es'))
            df['sentiment'] = df['text'].apply(generate_sentiment_analysis)
            df['emotion'] = df['text'].apply(generate_emotions_analysis)
            classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
            df['topics'] = df['text'].apply(classify_call)

        return df

    @reactive.calc
    def process_generic_webpage_for_df():
        if input.platform_selector() != "generic_webpage":
            return pd.DataFrame()
        
        url = input.generic_webpage_url()
        
        if not url:
            return pd.DataFrame({'Error': ["Por favor ingresa una URL"]})
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs_elements = soup.find_all('p')
            paragraphs = [p.get_text() for p in paragraphs_elements]
            if not paragraphs or (len(paragraphs)==1 and not  paragraphs[0].strip()):
                all_text = soup.get_text(separator='\n', strip=True)
                if all_text:
                    paragraphs=[line for line in all_text.splitlines() if line.strip()]
                if not paragraphs:
                    return pd.DataFrame({"Error": ["No se pudo encontrar texto"]})
            data = {
                'paragraph_number': range(1, len(paragraphs) + 1),
                'text': paragraphs,
                'length': [len(p) for p in paragraphs]#,
                #'sentiment': [generate_sentiment_analysis(p) for p in paragraphs],
                #'emotion': [generate_emotions_analysis(p) for p in paragraphs]
            }
            df = pd.DataFrame(data)
            testText = str(df['text'].iloc[0])
            collapsed_text = collapse_text(df)
            actual_candidate_labels = ['General']
            if collapsed_text:
                if not _ensure_topics_pipeline():
                    print("Error: Pipeline de tópicos no disponible")
                candidate_labels_str= topics_generator(collapsed_text)

                if isinstance(candidate_labels_str, str) and not candidate_labels_str.startswith("Error:"):
                    try:
                        parsed_labels = ast.literal_eval(candidate_labels_str)
                        if isinstance(parsed_labels, list):
                            actual_candidate_labels = parsed_labels
                        else:   
                            print(f"Error: La respuesta de Gemini no es una lista válida: {candidate_labels_str}")  
                    except (ValueError, SyntaxError) as e:
                        print(f"Error al procesar las etiquetas de tópicos: {e}")
                elif isinstance(candidate_labels_str, str) and candidate_labels_str.startswith("Error:"):
                    print(f"Error: La respuesta de Gemini no es una lista válida: {candidate_labels_str}")

            try: 
                languageDetected = detectLanguage(testText, api_key=DETECT_LANGUAGE_API_KEY)
            except Exception as e:
                languageDetected='es'
            #languageDetected = detectLanguage(testText, api_key=DETECT_LANGUAGE_API_KEY)
            if languageDetected=='es':
                df['sentiment'] = df['text'].apply(generate_sentiment_analysis)
                df['emotion'] = df['text'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['topics'] = df['text'].apply(classify_call)

            else: 
                df['text'] = df['text'].apply(lambda x: TranslateText(text=x, source= languageDetected, target='es'))
                df['sentiment'] = df['text'].apply(generate_sentiment_analysis)
                df['emotion'] = df['text'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['topics'] = df['text'].apply(classify_call)
            return df        
        except Exception as e:
            return pd.DataFrame({"Error": [f"Error al acceder a la página web: {str(e)}"]})



    @reactive.calc
    def getTweetsResponses():
        if input.platform_selector() != "twitter":
            return pd.DataFrame()
        
        #url = input.twitter_url()      
        #twitter_input = url
        #if not twitter_input  or not TWITTER_BEARER_TOKEN:
        #    return pd.DataFrame({'Error': ["URL de Twitter no válida o clave API no configurada."]})
        twitter_query = input.twitter_query().strip()
        if not twitter_query or not TWITTER_BEARER_TOKEN:
            return pd.DataFrame({"Error": ["Consulta de X antes Twitter no válida o Bearer Token mal configurado"]})

        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        tweets_list=[]
        response_data = None ##### Esto para poder poner un default

        ##### Variables comunes de los tweets 
        tweet_fields_to_request = ['created_at', 'public_metrics', 'author_id', 'conversation_id', 'in_reply_to_user_id', 'lang']
        expansions_to_request = ['author_id', 'in_reply_to_user_id']
        user_fields_to_request = ['username', 'name', 'profile_image_url', 'verified']
        max_results_count = 10
        try: 
            if("x.com/" in twitter_query and "/status/" in twitter_query)  or \
            ("x.com/" in twitter_query and "/status/" in twitter_query):
                match = re.search(r'.*/status/(\d+)', twitter_query)
                if match:
                    tweet_id = match.group(1)
                    print('Comenzará a buscar tweets/respuestas en una publicación')
                    response_data = client.search_recent_tweets(
                        query=f"conversation_id:{tweet_id}",
                        tweet_fields = tweet_fields_to_request,
                        expansions = expansions_to_request,
                        user_fields = user_fields_to_request,
                        max_results = max_results_count
                    )
                else: 
                   return pd.DataFrame({"Error": ["No se pudo extraer el ID del tweet."]})     
            elif twitter_query.startswith("#"):
                hashtag = twitter_query
                print('Comenzará a buscar tweets por hashtag')
                response_data = client.search_recent_tweets(
                    query = hashtag, 
                    tweet_fields = tweet_fields_to_request,
                    expansions = expansions_to_request,
                    user_fields = user_fields_to_request,
                    max_results = max_results_count
                    )
            elif twitter_query.startswith("@"):
                username_to_search = twitter_query.lstrip('@')
                print(f'Comenzará a buscar nombres con el nombre del usuario {username_to_search}')
                user_lookup = client.get_user(username=username_to_search, user_fields=['id'])
                if user_lookup.data:
                    user_id = user_lookup.data.id
                    response_data = client.get_users_tweets(
                        id = user_id,
                        tweet_fields = tweet_fields_to_request,
                        expansions = expansions_to_request,
                        user_fields = user_fields_to_request,
                        max_results = max_results_count
                    )
                else: 
                    return pd.DataFrame({"Error": [f'Información de usuario no encontrada {username_to_search}']})
            elif twitter_query.isdigit():
                tweet_id = twitter_query 
                print('Comenzará a buscar tweets por el tweet ID')
                response_data = client.search_recent_tweets(
                    query = f"conversation_id:{tweet_id}",
                    tweet_fields = tweet_fields_to_request,
                    expansions = expansions_to_request,
                    user_fields = user_fields_to_request,
                    max_results = max_results_count
                )
            else:
                username_to_search = twitter_query
                print(f"Buscando el nombre del usuario con la query {twitter_query}")
                user_lookup = client.get_user(username = username_to_search, user_fields=['id'])
                if user_lookup.data:
                    user_id = user_lookup.data.id
                    response_data = client.get_users_tweets(
                        id = user_id,
                        tweet_fields = tweet_fields_to_request,
                        expansions = expansions_to_request,
                        user_fields = user_fields_to_request,
                        max_results = max_results_count
                    )
                else: 
                    return pd.DataFrame({"Error": [f'Información de usuario no encontrada {username_to_search}']})
            if response_data and response_data.data:
                users_data = {u["id"]: u for u in response_data.includes.get('users', [])} if response_data.includes else {}
                for tweet in response_data.data:
                    author_info = users_data.get(tweet.author_id, {})
                    metrics = tweet.public_metrics if tweet.public_metrics else {}
                    tweet_info = {
                        'tweet_id': tweet.id,
                        'text': tweet.text,
                        'author_id': tweet.author_id,
                        'username': author_info.get("username", "N/A"),
                        'author_name': author_info.get("name", "N/A"),
                        'author_verified': author_info.get("verified", False),
                        'created_at': tweet.created_at,
                        'like_count': metrics.get('like_count', 0),
                        'retweet_count': metrics.get('retweet_count', 0),
                        'reply_count': metrics.get('reply_count', 0),
                        'quote_count': metrics.get('quote_count', 0),
                        'impression_count': metrics.get('impression_count', 0),
                        'conversation_id': tweet.conversation_id,
                        'in_reply_to_user_id': tweet.in_reply_to_user_id,
                        'lang': tweet.lang
                    }
                    tweets_list.append(tweet_info)
            
            if not tweets_list: 
                 return pd.DataFrame({'Mensaje': ["No se encontraron tweets para la consulta o la respuesta no contenía datos."]})

            df = pd.DataFrame(tweets_list)
            testText = str(df['text'].iloc[0])
            collapsed_text = collapse_text(df)
            actual_candidate_labels = ['General']
            if collapsed_text:
                if not _ensure_topics_pipeline():
                    print("Error: Pipeline de tópicos no disponible")
                candidate_labels_str= topics_generator(collapsed_text)

                if isinstance(candidate_labels_str, str) and not candidate_labels_str.startswith("Error:"):
                    try:
                        parsed_labels = ast.literal_eval(candidate_labels_str)
                        if isinstance(parsed_labels, list):
                            actual_candidate_labels = parsed_labels
                        else:   
                            print(f"Error: La respuesta de Gemini no es una lista válida: {candidate_labels_str}")  
                    except (ValueError, SyntaxError) as e:
                        print(f"Error al procesar las etiquetas de tópicos: {e}")
                elif isinstance(candidate_labels_str, str) and candidate_labels_str.startswith("Error:"):
                    print(f"Error: La respuesta de Gemini no es una lista válida: {candidate_labels_str}")

            try: 
                languageDetected = detectLanguage(testText, api_key=DETECT_LANGUAGE_API_KEY)
            except Exception as e:
                languageDetected='es'
            
            #languageDetected = detectLanguage(testText, api_key=DETECT_LANGUAGE_API_KEY)
            if languageDetected=="es":
                df['sentiment'] = df['text'].apply(generate_sentiment_analysis)
                df['emotion'] = df['text'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['topics'] = df['text'].apply(classify_call)

            else: 
                df['text'] = df['text'].apply(lambda x: TranslateText(text=x, source= languageDetected, target='es')) 
                df['sentiment'] = df['text'].apply(generate_sentiment_analysis)
                df['emotion'] = df['text'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['topics'] = df['text'].apply(classify_call)
            
            return df
        
        except tweepy.TweepyException as e:
            error_message = str(e)
            if hasattr(e, 'api_errors') and e.api_errors and isinstance(e.api_errors, list) and e.api_errors[0]:
                api_error = e.api_errors[0]
                if isinstance(api_error, dict): 
                    error_message = api_error.get('detail', error_message)
                    if 'title' in api_error: error_message = f"{api_error['title']}: {error_message}"
                elif hasattr(api_error, 'message'): # Old format
                     error_message = api_error.message
            elif hasattr(e, 'response') and e.response is not None:
                 try:
                     error_details = e.response.json()
                     error_message = error_details.get('detail', error_details.get('title', str(e)))
                 except ValueError: # Not a JSON response
                     error_message = e.response.text if e.response.text else str(e)
            
            print(f"TweepyException: {error_message}")
            return pd.DataFrame({"Error": [f"Error de API de Twitter: {error_message}"]})
        ### Excepciones basadas en la API
        ### Exexpciones no basadas en la API

        #if "x.com/" in twitter_input and "/status/" in twitter_input:
        #    #match = re.search(r'/status/(\d+)/', twitter_input)
        #    match = re.match(r'.*/status/(\d+)', twitter_input)
        #    if match:
        #        tweet_id = match.group(1)
        #    else:
        #        return pd.DataFrame({'Error': ["No se pudo extraer el ID del tweet."]})
        #elif twitter_input.isdigit():
        #    tweet_id = twitter_input
        #else: 
        #    return pd.DataFrame({'Error': ["URL de Twitter no válida."]}) 
        #     
        #try: 
        #    listTweets = client.search_recent_tweets(
        #        query=f"conversation_id:{tweet_id}",
        #        expansions=["author_id"],  
        #        user_fields=["username"],  
        #        max_results=10
        #    )
        #
        #    tweets = listTweets.data
        #    users = {u["id"]: u for u in listTweets.includes['users']}
        #    tweets_list = []
        #    for tweet in tweets:
        #        tweet_info = {
        #            'tweet_id': tweet.id,
        #            'text': tweet.text,
        #            'author_id': tweet.author_id,
        #            'username': users[tweet.author_id]["username"] if tweet.author_id in users else None,
        #            'created_at': tweet.created_at if hasattr(tweet, 'created_at') else None
        #        }
        #        tweets_list.append(tweet_info)
        #    df = pd.DataFrame(tweets_list)
        #    df['sentiment'] = df['text'].apply(generate_sentiment_analysis)
        #    #df['emotion'] = df['text'].apply(lambda x: detectEmotion(x)[0])
        #    #return pd.DataFrame(df)
        #    return df 
        except Exception as e:
            #return pd.DataFrame({"Error": [f"Error al obtener comentarios: {str(e)}"]})
            print(f"Error general en la función getTweetsResponses: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame({"Error": [f"Error general en la función getTweetsResponses: {str(e)}"]})


    
    @reactive.calc
    def get_youtube_comments():
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        if input.platform_selector() != "youtube":
            return pd.DataFrame()
        video_url = input.youtube_url()
        
        if "v=" in video_url:
            video_id = video_url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[-1].split("?")[0]
        if not video_id or not YOUTUBE_API_KEY:
        #if not video_id or not youtube_api_key:
            return pd.DataFrame({'Error': ["URL de Youtube no válida o clave API no configurada."]})
        try:
            comments = []
            response = youtube.commentThreads().list(
                part='snippet', 
                videoId=video_id,
                textFormat='plainText',
                maxResults=10
            ).execute()
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                comments.append({'author': author, 'comment': comment})
            ### Esto será comentado, pero por ahora solo quiero los primeros 10 comentarios
            #next_page_token = None
            #while True: 
            #    response = youtube.commentThreads().list(
            #        part='snippet',
            #        videoId=video_id,
            #        textFormat='plainText',
            #        pageToken=next_page_token,
            #        maxResults=100
            #    ).execute()
            #    for item in response['items']:
            #        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            #        author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            #        comments.append({'author': author, 'comment': comment})
            #    next_page_token = response.get('nextPageToken')
            #    if not next_page_token:
            #        break
            df = pd.DataFrame(comments)
            testText = str(df['comment'].iloc[0])
            collapsed_text = collapse_text(df)
            actual_candidate_labels = ['General']
            if collapsed_text:
                if not _ensure_topics_pipeline():
                    print("Error: Pipeline de tópicos no disponible")
                candidate_labels_str= topics_generator(collapsed_text)

                if isinstance(candidate_labels_str, str) and not candidate_labels_str.startswith("Error:"):
                    try:
                        parsed_labels = ast.literal_eval(candidate_labels_str)
                        if isinstance(parsed_labels, list):
                            actual_candidate_labels = parsed_labels
                        else:   
                            print(f"Error: La respuesta de Gemini no es una lista válida: {candidate_labels_str}")  
                    except (ValueError, SyntaxError) as e:
                        print(f"Error al procesar las etiquetas de tópicos: {e}")
                elif isinstance(candidate_labels_str, str) and candidate_labels_str.startswith("Error:"):
                    print(f"Error: La respuesta de Gemini no es una lista válida: {candidate_labels_str}")

            try: 
                languageDetected = detectLanguage(testText, api_key=DETECT_LANGUAGE_API_KEY)
            except Exception as e:
                languageDetected='es'
            #languageDetected = detectLanguage(testText, api_key=DETECT_LANGUAGE_API_KEY)
            if languageDetected=="es":
                df['sentiment'] = df['comment'].apply(generate_sentiment_analysis)
                df['emotion'] = df['comment'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['topics'] = df['comment'].apply(classify_call)

            else: 
                df['comment'] = df['comment'].apply(lambda x: TranslateText(text=x, source= languageDetected, target='es')) 
                df['sentiment'] = df['comment'].apply(generate_sentiment_analysis)
                df['emotion'] = df['comment'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['topics'] = df['comment'].apply(classify_call)

            #df['sentiment'] = df['comment'].apply(generate_sentiment_analysis)
            #df['emotion'] = df['comment'].apply(generate_emotions_analysis)
            #df['emotion'] = df['comment'].apply(lambda x: detectEmotion(x)[0])
            return df
        except Exception as e:
            return pd.DataFrame({"Error": [f"Error al obtener comentarios: {str(e)}"]})


    @reactive.calc
    def mapsComments():
        if input.platform_selector() != "maps":
            return pd.DataFrame()
        placename = input.maps_query()
        if not placename or not MAPS_API_KEY:
            map_coordinates.set(None)
            return 'Falta el nombre del lugar o no hay una clave válida'
        try:
            gmaps = googlemaps.Client(MAPS_API_KEY)
            find_place_result = gmaps.find_place(placename, input_type='textquery')
            comments = []

            if find_place_result['status'] == 'OK' and find_place_result.get('candidates'):
                place_id_from_find = find_place_result['candidates'][0]['place_id']
                
                # Fetch details including geometry and the canonical place_id
                place_details = gmaps.place(place_id_from_find,
                                            fields=['name', 'rating', 'review', 'formatted_address', 'geometry', 'place_id'],
                                            language='es')

                current_map_data = {}
                if place_details.get('result'):
                    result_data = place_details['result']
                    # Use the place_id from the place details response, as it's more authoritative
                    current_map_data['place_id'] = result_data.get('place_id', place_id_from_find)
                    
                    if result_data.get('geometry') and result_data['geometry'].get('location'):
                        location = result_data['geometry']['location']
                        current_map_data['lat'] = location['lat']
                        current_map_data['lng'] = location['lng']
                    
                    if 'place_id' in current_map_data or ('lat' in current_map_data and 'lng' in current_map_data):
                        map_coordinates.set(current_map_data)
                    else:
                        map_coordinates.set(None) # Not enough info from place details

                    reviews_data = result_data.get('reviews', [])
                    # Process reviews if any
                    if reviews_data: # Check if reviews_data is not None
                        # The original code had a loop for next_page_token which is complex here.
                        # For simplicity, this version processes only the first batch of reviews.
                        # If pagination of reviews is critical, that logic would need to be reinstated carefully.
                        for review in reviews_data:
                            comments.append({'author': review.get('author_name', 'N/A'), 
                                             'comment': review.get('text', ''), 
                                             'rating': review.get('rating', 'N/A')})
                else: # place_details call failed or no result
                    map_coordinates.set(None)
            else: # find_place_result failed or no candidates
                map_coordinates.set(None)

            df = pd.DataFrame(comments)
            testText = str(df['comment'].iloc[0])
            collapsed_text = collapse_text(df)
            actual_candidate_labels = ['General']
            if collapsed_text:
                if not _ensure_topics_pipeline():
                    print("Error: Pipeline de tópicos no disponible")
                candidate_labels_str= topics_generator(collapsed_text)

                if isinstance(candidate_labels_str, str) and not candidate_labels_str.startswith("Error:"):
                    try:
                        parsed_labels = ast.literal_eval(candidate_labels_str)
                        if isinstance(parsed_labels, list):
                            actual_candidate_labels = parsed_labels
                        else:   
                            print(f"Error: La respuesta de Gemini no es una lista válida: {candidate_labels_str}")  
                    except (ValueError, SyntaxError) as e:
                        print(f"Error al procesar las etiquetas de tópicos: {e}")
                elif isinstance(candidate_labels_str, str) and candidate_labels_str.startswith("Error:"):
                    print(f"Error: La respuesta de Gemini no es una lista válida: {candidate_labels_str}")

            try: 
                languageDetected = detectLanguage(testText, api_key=DETECT_LANGUAGE_API_KEY)
            except Exception as e:
                languageDetected='es'           
            #languageDetected = detectLanguage(testText, api_key=DETECT_LANGUAGE_API_KEY)
            if languageDetected=="es":
                df['sentiment'] = df['comment'].apply(generate_sentiment_analysis)
                df['emotion'] = df['comment'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['topics'] = df['comment'].apply(classify_call)
            else: 
                df['comment'] = df['comment'].apply(lambda x: TranslateText(text=x, source= languageDetected, target='es')) 
                df['sentiment'] = df['comment'].apply(generate_sentiment_analysis)
                df['emotion'] = df['comment'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['topics'] = df['comment'].apply(classify_call)
            
            #df['sentiment'] = df['comment'].apply(generate_sentiment_analysis)
            #df['emotion'] = df['comment'].apply(generate_emotions_analysis)

            #df['emotion'] = df['comment'].apply(lambda x: detectEmotion(x)[0])
            return df
        except Exception as e: 
            return pd.DataFrame({"Error": [f"Error al obtener comentarios: {str(e)}"]})


    @reactive.calc
    def get_reddit_comments():
        if input.platform_selector() != "reddit":
            return pd.DataFrame()
        reddit_url = input.reddit_url()
        if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
            return pd.DataFrame({"Error": ["Reddit API credentials not configured in environment variables."]})

        try:
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
                read_only=True
            )

            # Extract submission ID from URL (more robust extraction might be needed for all URL formats)
            match = re.search(r"comments/([a-zA-Z0-9]+)/?", reddit_url)
            if not match:
                return pd.DataFrame({"Error": ["Invalid Reddit submission URL format. Could not extract submission ID."]})
            
            submission_id = match.group(1)
            submission = reddit.submission(id=submission_id)
            
            #print(f"Fetching comments for submission ID: {submission_id} (Title: {submission.title})")

            # Fetch all comments, replacing MoreComments objects
            # This can take time for posts with many comments
            submission.comments.replace_more(limit=None) # limit=0 for all, limit=None is default for all top-level
                                                        # For very large threads, consider a limit or iterative fetching.

            comments_data = []
            for comment in submission.comments.list(): # .list() flattens the comment tree
                if isinstance(comment, praw.models.Comment): # Ensure it's a Comment object
                    comments_data.append({
                        'id': comment.id,
                        'original_url': reddit_url, 
                        'reddit_title': submission.title, 
                        'author': str(comment.author) if comment.author else "[deleted]",
                        'comment': comment.body,
                        'score': comment.score,
                        'created_utc': pd.to_datetime(comment.created_utc, unit='s'),
                        'parent_id': comment.parent_id, # ID of the parent (submission or another comment)
                        'permalink': f"https://www.reddit.com{comment.permalink}",
                        'is_submitter': comment.is_submitter,
                        'edited': False if isinstance(comment.edited, bool) else pd.to_datetime(comment.edited, unit='s'), # PRAW returns False or timestamp
                        'depth': comment.depth
                    })
            
            if not comments_data:
                return pd.DataFrame({"Mensaje": ["No comments found for this submission or comments are not public."]})

            df = pd.DataFrame(comments_data)
            testText = str(df['comment'].iloc[0])
            collapsed_text = collapse_text(df)
            actual_candidate_labels = ['General']
            if collapsed_text:
                if not _ensure_topics_pipeline():
                    print("Error: Pipeline de tópicos no disponible")
                candidate_labels_str= topics_generator(collapsed_text)

                if isinstance(candidate_labels_str, str) and not candidate_labels_str.startswith("Error:"):
                    try:
                        parsed_labels = ast.literal_eval(candidate_labels_str)
                        if isinstance(parsed_labels, list):
                            actual_candidate_labels = parsed_labels
                        else:   
                            print(f"Error: La respuesta de Gemini no es una lista válida: {candidate_labels_str}")  
                    except (ValueError, SyntaxError) as e:
                        print(f"Error al procesar las etiquetas de tópicos: {e}")
                elif isinstance(candidate_labels_str, str) and candidate_labels_str.startswith("Error:"):
                    print(f"Error: La respuesta de Gemini no es una lista válida: {candidate_labels_str}")

            try: 
                languageDetected = detectLanguage(testText, api_key=DETECT_LANGUAGE_API_KEY)
            except Exception as e:
                languageDetected='es'

            #languageDetected = detectLanguage(testText, api_key=DETECT_LANGUAGE_API_KEY)
            if languageDetected=="es":
                df['sentiment'] = df['comment'].apply(generate_sentiment_analysis)
                df['emotion'] = df['comment'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['topics'] = df['comment'].apply(classify_call)
            
            else: 
                df['comment'] = df['comment'].apply(lambda x: TranslateText(text=x, source= languageDetected, target='es')) 
                df['sentiment'] = df['comment'].apply(generate_sentiment_analysis)
                df['emotion'] = df['comment'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['topics'] = df['comment'].apply(classify_call)

            #df['sentiment'] = df['comment'].apply(generate_sentiment_analysis)
            #df['emotion'] = df['comment'].apply(generate_emotions_analysis)
            return df

        except praw.exceptions.PRAWException as e:
            print(f"PRAW API Error: {e}")
            return pd.DataFrame({"Error": [f"Reddit API error: {str(e)}"]})
        except requests.exceptions.RequestException as e: # PRAW uses requests
            print(f"Network Error: {e}")
            return pd.DataFrame({"Error": [f"Network error while contacting Reddit: {str(e)}"]})
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame({"Error": [f"An unexpected error occurred: {str(e)}"]})

    @reactive.calc
    def get_playstore_comments():
        if input.platform_selector() != "playstore":
            return pd.DataFrame()
        try:
            playstore_url = str(input.playstore_url())
            if re.match(r'.*search\?q=(.+?)&', playstore_url):
                search_string = re.search(r'search\?q=(.+?)&', playstore_url).group(1) 
                result = search(
                    search_string,
                    lang="es",  
                    country="mx",  
                    n_hits=1
                    )
                app_id = result[0]['appId']
                reviews = play_reviews(
                    app_id,
                    lang='es', 
                    country='mx', 
                    sort=Sort.NEWEST, 
                    count=30, 
                    filter_score_with=None 
                )
            else:            
                app_id = re.search(r'(?<=id=)[^&]+', playstore_url).group(0)
                reviews = play_reviews(
                    app_id,
                    lang='es', 
                    country='mx', 
                    sort=Sort.NEWEST, 
                    count=30, 
                    filter_score_with=None 
                )
            df = pd.DataFrame(reviews[0])
            testText = str(df['content'].iloc[0])
            collapsed_text = collapse_text(df)
            actual_candidate_labels = ['General']
            if collapsed_text:
                if not _ensure_topics_pipeline():
                    print("Error: Pipeline de tópicos no disponible")
                candidate_labels_str= topics_generator(collapsed_text)

                if isinstance(candidate_labels_str, str) and not candidate_labels_str.startswith("Error:"):
                    try:
                        parsed_labels = ast.literal_eval(candidate_labels_str)
                        if isinstance(parsed_labels, list):
                            actual_candidate_labels = parsed_labels
                        else:   
                            print(f"Error: La respuesta de Gemini no es una lista válida: {candidate_labels_str}")  
                    except (ValueError, SyntaxError) as e:
                        print(f"Error al procesar las etiquetas de tópicos: {e}")
                elif isinstance(candidate_labels_str, str) and candidate_labels_str.startswith("Error:"):
                    print(f"Error: La respuesta de Gemini no es una lista válida: {candidate_labels_str}")

            try: 
                languageDetected = detectLanguage(testText, api_key=DETECT_LANGUAGE_API_KEY)
            except Exception as e:
                languageDetected='es'
            
            #languageDetected = detectLanguage(testText, api_key=DETECT_LANGUAGE_API_KEY)
            if languageDetected=="es":
                df['sentiment'] = df['content'].apply(generate_sentiment_analysis)
                df['emotion'] = df['content'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['topics'] = df['content'].apply(classify_call)
            else: 
                df['content'] = df['content'].apply(lambda x: TranslateText(text=x, source= languageDetected, target='es')) 
                df['sentiment'] = df['content'].apply(generate_sentiment_analysis)
                df['emotion'] = df['content'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['topics'] = df['content'].apply(classify_call)
            return df     
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame({"Error": [f"An unexpected error occurred: {str(e)}"]})



    ## Iniciando la función del LLM
    def _ensure_llm_model():
        if llm_model_instance.get() is None:
            if OPENROUTER_API_KEY:
                try:
                    print('Iniciando el modelo LLM')
                    model = LangchainChatOpenAI(
                        openai_api_key = OPENROUTER_API_KEY,
                        model ="deepseek/deepseek-chat-v3-0324:free", 
                        base_url="https://openrouter.ai/api/v1"
                    )
                    llm_model_instance.set(model)
                    return True
                except Exception as e:
                    print(f"Error al cargar el modelo para LLM de DeepSeek {e}") 
                    return False 
            else: 
                print("Error: Clave de OpenRouter no configurada")
                return False        
        return llm_model_instance.get() is not None
    
    parse_template = (
        "You are tasked with extracting specific information from the following text content: {dom_content}. "
        "Please follow these instructions carefully:\n\n"
        "1. **Task:** Extract data from the provided text that matches the description: {parse_description}.\n"
        "2. **Output Format:** Return the extracted data ONLY as one or more Markdown tables. Each table MUST be correctly formatted.\n"
        "3. **Markdown Table Format:** Each table must adhere to the following Markdown format:\n"
        "   - Start with a header row, clearly labeling each column, separated by pipes (|).\n"
        "   - Follow the header row with an alignment row, using hyphens (-) to indicate column alignment (e.g., --- for left alignment).\n"
        "   - Subsequent rows should contain the data, with cells aligned according to the alignment row.\n"
        "   - Use pipes (|) to separate columns in each data row.\n"
        "4. **No Explanations:** Do not include any introductory or explanatory text before or after the table(s).\n"
        "5. **Empty Response:** If no information matches the description, return an empty string ('').\n"
        "6. **Multiple Tables:** If the text contains multiple tables matching the description, return each table separately, following the Markdown format for each.\n"
        "7. **Accuracy:** The extracted data must be accurate and reflect the information in the provided text.\n"
    )

    parse_prompt_template = ChatPromptTemplate.from_template(parse_template)

    # Function to parse and extract information from the chunks
    def parse_content_with_llm(cleaned_contents, parse_description):
        if not _ensure_llm_model():
            return ["Error: Modelo LLM no disponible para parseo."]
        model = llm_model_instance.get()
        
        all_raw_outputs = []
        for i, content in enumerate(cleaned_contents):
            if content.startswith("Error scraping"):
                all_raw_outputs.append(f"Error processing content from URL {i+1}: {content}")
                continue
            
            dom_chunks = split_dom_content(content)
            
            for j, chunk in enumerate(dom_chunks):
                try:
                    response = model.invoke(parse_prompt_template.format_prompt(dom_content=chunk, parse_description=parse_description))
                    all_raw_outputs.append(response.content)
                    print(f"Parsed chunk {j+1} from content {i+1}")
                except Exception as e:
                    print(f"Error parsing chunk {j+1} from content {i+1}: {e}")
                    all_raw_outputs.append(f"Error parsing chunk {j+1} from content {i+1}: {e}")

        return all_raw_outputs


    def extract_tables_from_llm_outputs(raw_outputs_list):
        all_dfs = []
        if not isinstance(raw_outputs_list, list):
            print(f"Error: extract_tables_from_llm_outputs expected a list, got {type(raw_outputs_list)}")
            return all_dfs
            
        for raw_output_md in raw_outputs_list:
            if isinstance(raw_output_md, str) and not raw_output_md.startswith("Error processing content") and not raw_output_md.startswith("Error parsing chunk"):
                dfs_from_chunk = markdown_to_csv(raw_output_md)
                if dfs_from_chunk: # markdown_to_csv returns a list of DFs
                    all_dfs.extend(dfs_from_chunk)
        return all_dfs

    
    #@reactive.calc
    def plot_sentiment_distribution_plotly(df):
        sentiment_categories = ['Positivo', 'Neutral', 'Negativo']
        df['sentiment'] = pd.Categorical(df['sentiment'], categories=sentiment_categories, ordered=True)
        colors = {
            'Positivo': '#0d9a66',  # Verde
            'Neutral': '#ffe599',   # Amarillou XD
            'Negativo': '#ff585d'   # Rojo
        }
        grouped_df = df.groupby('sentiment', observed=False).size().reset_index(name='counts')
        color_list = [colors[cat] for cat in grouped_df['sentiment']]
        barplot = px.bar(grouped_df, x='sentiment', y='counts', title='Sentiment Analysis of the paragraphs', color='sentiment', 
                        color_discrete_sequence=color_list, category_orders={'sentiment': sentiment_categories})
        return barplot
        #return go.FigureWidget(barplot)

    def plot_sentiment_distribution_seaborn(df_input: pd.DataFrame):
        plt.style.use('seaborn-v0_8-darkgrid') # Using a seaborn style that complements dark themes

        sentiment_categories = ['Positivo', 'Neutral', 'Negativo']
        # Ensure the sentiment column is categorical for correct ordering and handling of all categories
        df_input['sentiment'] = pd.Categorical(df_input['sentiment'], categories=sentiment_categories, ordered=True)
        
        sentiment_counts = df_input['sentiment'].value_counts().reindex(sentiment_categories, fill_value=0)
        
        # Define a color palette (you can choose others from Seaborn)
        # Using a palette that generally works well with dark backgrounds
        palette = {"Positivo": "#2ECC71", "Neutral": "#F1C40F", "Negativo": "#E74C3C"}
        # Ensure colors are in the order of sentiment_counts.index
        bar_colors = [palette.get(s, '#cccccc') for s in sentiment_counts.index]

        fig, ax = plt.subplots(figsize=(7, 5)) # Adjusted figure size
        
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=bar_colors, ax=ax, width=0.6)
        
        ax.set_title('Análisis de Sentimiento de los Comentarios', fontsize=16, pad=20)
        ax.set_xlabel('Sentimiento', fontsize=14, labelpad=15)
        ax.set_ylabel('Número de Comentarios', fontsize=14, labelpad=15)
        ax.tick_params(axis='both', which='major', labelsize=12)
        total_sentiments = sentiment_counts.sum()
        
        # Add value annotations on top of bars
        for i, count in enumerate(sentiment_counts.values):
            if count > 0: 
                percentage = (count / total_sentiments) * 100
                annotation_text = f"{count} ({percentage:.1f}%)"
                ax.text(i, count + (sentiment_counts.max() * 0.015 if sentiment_counts.max() > 0 else 0.15), 
                        annotation_text, ha='center', va='bottom', fontsize=9, color='black')
        
        plt.tight_layout()
        return fig




    def plot_topics_distribution_seaborn(df_input: pd.DataFrame):
        plt.style.use('seaborn-v0_8-darkgrid') 

        topics_counts = df_input['topics'].value_counts()

        fig, ax = plt.subplots(figsize=(7, 5)) 


        #sns.barplot(x=topics_counts.index, y=topics_counts.values, ax=ax, width=0.6, palette="viridis", orient='y') 
        sns.barplot(x=topics_counts.index, y=topics_counts.values, ax=ax, width=0.6, palette="viridis") 
        
        ax.set_title('Análisis de Tópicos de los Comentarios', fontsize=16, pad=20)
        ax.set_xlabel('Tópicos', fontsize=14, labelpad=15)
        ax.set_ylabel('Número de Comentarios', fontsize=14, labelpad=15)
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.xticks(rotation=45, ha="right") 
        
        total_topics = topics_counts.sum()
        
        # Add value annotations on top of bars
        # Corrected to use topics_counts instead of emotion_counts
        for i, count in enumerate(topics_counts.values):
            if count > 0: 
                percentage = (count / total_topics) * 100 if total_topics > 0 else 0
                annotation_text = f"{count} ({percentage:.1f}%)"
                ax.text(i, count + (topics_counts.max() * 0.015 if topics_counts.max() > 0 else 0.15), 
                        annotation_text, ha='center', va='bottom', fontsize=9, color='black')
        
        plt.tight_layout()
        return fig

    def plot_emotion_distribution_seaborn(df_input: pd.DataFrame):
        plt.style.use('seaborn-v0_8-darkgrid') 
        emotion_categories_es = ["Alegría", "Tristeza", "Enojo", "Miedo", "Sorpresa", "Asco", "Neutral", "Desconocida", "Error en análisis"]
        df_input['emotion'] = pd.Categorical(df_input['emotion'], categories=emotion_categories_es, ordered=True)
        
        emotion_counts = df_input['emotion'].value_counts().reindex(emotion_categories_es, fill_value=0)
        
        emotion_palette_es = {
            "Alegría": "#4CAF50",   # Green
            "Tristeza": "#2196F3",  # Blue
            "Enojo": "#F44336",     # Red
            "Miedo": "#9C27B0",     # Purple
            "Sorpresa": "#FFC107",  # Amber
            "Asco": "#795548",      # Brown
            "Neutral": "#9E9E9E",   # Grey
            "Desconocida": "#607D8B", # Blue Grey
            "Error en análisis": "#BDBDBD" # Light Grey
        }
        bar_colors = [emotion_palette_es.get(cat, '#cccccc') for cat in emotion_counts.index]

        fig, ax = plt.subplots(figsize=(7, 5)) # Adjusted figure size
 
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette=bar_colors, ax=ax, width=0.6)
        
        ax.set_title('Análisis de Emociones de los Comentarios', fontsize=16, pad=20)
        ax.set_xlabel('Emoción', fontsize=14, labelpad=15)
        ax.set_ylabel('Número de Comentarios', fontsize=14, labelpad=15)
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.xticks(rotation=45, ha="right") # Rotate labels if they overlap
        total_emotions = emotion_counts.sum()
        
        # Add value annotations on top of bars
        for i, count in enumerate(emotion_counts.values):
            if count > 0: 
                percentage = (count / total_emotions) * 100
                annotation_text = f"{count} ({percentage:.1f}%)"
                ax.text(i, count + (emotion_counts.max() * 0.015 if emotion_counts.max() > 0 else 0.15), 
                        annotation_text, ha='center', va='bottom', fontsize=9, color='black')
        

        plt.tight_layout()
        return fig 

    #@reactive.calc
    def plot_emotion_distribution(df):
        grouped_df= df.groupby('emotion', observed=False).size().reset_index(name='counts')
        barplot = px.bar(grouped_df, x='emotion', y='counts', title='Emotion Analysis of the paragraphs', 
                         color='emotion')
        #return go.FigureWidget(barplot)
        return barplot    

    
    #@output
    #@render.text 
    #def summary_output():
    #    #platform = input.platform_selector()
    #    #if platform == "wikipedia":
    #    #    df=  process_wikipedia_for_df()
    #    #elif platform == "youtube":
    #    #    df= get_youtube_comments()
    #    #elif platform == "maps":
    #    #    df= mapsComments()
    #    #elif platform =='twitter':
    #    #    df= getTweetsResponses()
    #    #else: 
    #    #    return "Selecciona plataforma válida"
    #    df = processed_dataframe.get()
    #    platform = input.platform_selector()
    #    #if df.empty:
    #    if not isinstance(df, pd.DataFrame) or df.empty:
    #        return "No hay datos disponibles para resumir."
    #    #if 'Error' in df.columns:
    #    #    return "No se puede generar un resumen debido a un error previo"
    #    if ('Error' in df.columns and len(df) == 1) or \
    #       ('Mensaje' in df.columns and len(df) == 1) :
    #        return "No se puede generar un resumen a partir de un mensaje de error o un mensaje informativo."        
    #
    #    text_to_summarize = collapse_text(df)
    #    #return summary_generator(text_to_summarize)
    #    return summary_generator(text_to_summarize, platform)


    #@output
    @reactive.calc
    def calculate_summary_text():
        df = processed_dataframe.get()
        platform = input.platform_selector()
        #if df.empty:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return "No hay datos disponibles para resumir."
        #if 'Error' in df.columns:
        #  return "No se puede generar un resumen debido a un error previo"
        if ('Error' in df.columns and len(df) == 1) or \
           ('Mensaje' in df.columns and len(df) == 1) :
            return "No se puede generar un resumen a partir de un mensaje de error o un mensaje informativo."        

        text_to_summarize = collapse_text(df)
        #return summary_generator(text_to_summarize)
        return summary_generator(text_to_summarize, platform)


    @reactive.calc
    def calculate_topic_text():
        df = processed_dataframe.get()
        if not isinstance(df, pd.DataFrame) or df.empty:
            return "No hay datos disponibles para resumir."
        if ('Error' in df.columns and len(df) == 1) or \
           ('Mensaje' in df.columns and len(df) == 1) :
            return "No se puede generar un resumen a partir de un mensaje de error o un mensaje informativo."        
        text = collapse_text(df)
        return topics_generator(text)


    @reactive.calc
    def calculate_overall_topic_text():
        df = processed_dataframe.get()
        if not isinstance(df, pd.DataFrame) or df.empty:
            return "No hay datos disponibles para resumir."
        if ('Error' in df.columns and len(df) == 1) or \
           ('Mensaje' in df.columns and len(df) == 1) :
            return "No se puede generar un resumen a partir de un mensaje de error o un mensaje informativo."        
        text = collapse_text(df)
        return topics_generator(text)


    @output
    @render.ui
    def styled_summary_output():
        summary_text = calculate_summary_text() # Get text from our reactive calc

        if not summary_text or summary_text == "No hay datos disponibles para resumir." or \
           summary_text == "No se puede generar un resumen a partir de un mensaje de error o un mensaje informativo." or \
           summary_text.startswith("Error:"):
            return ui.card(
                ui.card_header(
                    ui.tags.h5("Resumen", class_="card-title")
                ),
                ui.markdown(f"_{summary_text}_") # Italicize messages/errors
            )

        # Try to split title and body for better formatting
        parts = summary_text.split(":\n", 1)
        summary_title = "Resumen"
        summary_body = summary_text
        if len(parts) == 2:
            summary_title = parts[0]
            summary_body = parts[1]

        return ui.card(
            ui.card_header(ui.tags.h5(summary_title, class_="card-title", style="margin-bottom: 0;")),
            ui.markdown(summary_body)
        )

    # '''Generar la red '''  
    @reactive.calc
    def generate_mind_map_data_llm():
        df = processed_dataframe.get()
        if not isinstance(df, pd.DataFrame) or df.empty:
            return "No hay datos para generar el mapa mental."
        if ('Error' in df.columns and len(df) == 1) or \
           ('Mensaje' in df.columns and len(df) == 1):
            return "No se puede generar un mapa mental a partir de un mensaje de error o informativo."

        text_for_mind_map = collapse_text(df)
        if not text_for_mind_map or text_for_mind_map.startswith("Error"):
            return "Texto no válido para generar el mapa mental."

        if not _ensure_gemini_model():
            return "Error: Modelo de Gemini no disponible para el mapa mental."
        
        gemini_model = gemini_model_instance.get()
        max_gemini_input_len = 7000 # Ajusta según sea necesario
        if len(text_for_mind_map) > max_gemini_input_len:
            text_to_process = text_for_mind_map[:max_gemini_input_len]
            print(f"Texto para mapa mental (Gemini) truncado a {max_gemini_input_len} caracteres.")
        else:
            text_to_process = text_for_mind_map

        mind_map_prompt = (
            f"Analiza el siguiente texto y extrae los conceptos clave y sus relaciones para construir un mapa mental.\n"
            f"Texto:\n---\n{text_to_process}\n---\n"
            f"Identifica un concepto central. Luego, identifica los conceptos principales que se derivan de él, y sub-conceptos si es aplicable.\n"
            f"Formatea tu respuesta EXCLUSIVAMENTE como un objeto JSON. El JSON debe tener una clave 'nodes' (una lista de objetos, cada uno con 'id' y 'label') y una clave 'edges' (una lista de objetos, cada uno con 'from' y 'to', refiriéndose a los ids de los nodos).\n"
            f"Ejemplo de formato JSON esperado:\n"
            f"{{\n"
            f'  "nodes": [\n'
            f'    {{"id": 1, "label": "Concepto Central"}},\n'
            f'    {{"id": 2, "label": "Idea Principal A"}},\n'
            f'    {{"id": 3, "label": "Sub-idea A1"}},\n'
            f'    {{"id": 4, "label": "Idea Principal B"}}\n'
            f'  ],\n'
            f'  "edges": [\n'
            f'    {{"from": 1, "to": 2}},\n'
            f'    {{"from": 2, "to": 3}},\n'
            f'    {{"from": 1, "to": 4}}\n'
            f'  ]\n'
            f"}}\n"
            f"Asegúrate de que el JSON sea válido. No incluyas NINGÚN texto, explicación, ni markdown antes o después del JSON."
        )
        try:
            response = gemini_model.generate_content(mind_map_prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error al generar datos para el mapa mental con Gemini: {str(e)}"

    # ''' Análisis de sentimiento'''  
    @output
    #@render_plotly
    @render.plot
    def sentiment_output():
        #platform = input.platform_selector()
        #if platform == "wikipedia":
        #    df = process_wikipedia_for_df()
        #elif platform == "twitter":
        #    df = getTweetsResponses()
        #elif platform == "youtube":
        #    df = get_youtube_comments()
        #elif platform == "maps":
        #    df = mapsComments()
        #else:
        #    return None
        df = processed_dataframe.get()
        #if df.empty or 'sentiment' not in df.columns:
        #    fig = px.bar(title='Análisis de Sentimiento (Sin datos)')
        #    fig.update_layout(xaxis_title="Sentimiento", yaxis_title="Conteo")
        #    return fig
        #    #return go.FigureWidget(fig)
        if not isinstance(df, pd.DataFrame) or df.empty or 'sentiment' not in df.columns:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(10,7))
            ax.text(0.5, 0.5, 'Análisis de Sentimiento (Sin datos)', 
                    ha='center', va='center', fontsize=14, color='white')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.patch.set_facecolor('#222222') # Match dark theme background
            ax.set_facecolor('#222222')
            plt.tight_layout()
            return fig
        if 'Error' in df.columns:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(10,7))
            error_message = df['Error'].iloc[0]
            ax.text(0.5, 0.5, f"Análisis de Sentimiento\n(Error: {error_message})", 
                    ha='center', va='center', fontsize=14, color='white', wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.patch.set_facecolor('#222222') # Match dark theme background
            ax.set_facecolor('#222222')
            plt.tight_layout()
        #    fig = px.bar(title=f"Análisis de Sentimiento (Error: {df['Error'].iloc[0]})")
        #    return fig
        #    #return go.FigureWidget(fig)        
        #return plot_sentiment_distribution(df)
        return plot_sentiment_distribution_seaborn(df)

    @render.download(filename='analisis_sentimientos.png')
    async def download_sentiment_plot():
        df = processed_dataframe.get()
        if not isinstance(df, pd.DataFrame) or df.empty or 'sentiment' not in df.columns:
            # Create a simple text image for "No data"
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(6,4))
            ax.text(0.5, 0.5, 'Análisis de Sentimiento\n(Sin datos para graficar)', 
                    ha='center', va='center', fontsize=12, color='white')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.patch.set_facecolor('#222222')
            ax.set_facecolor('#222222')
            plt.tight_layout()
        elif 'Error' in df.columns and len(df) == 1:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(6,4))
            error_message = df['Error'].iloc[0]
            ax.text(0.5, 0.5, f"Análisis de Sentimiento\n(Error: {error_message})", 
                    ha='center', va='center', fontsize=12, color='white', wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.patch.set_facecolor('#222222')
            ax.set_facecolor('#222222')
            plt.tight_layout()
        else:
            fig = plot_sentiment_distribution_seaborn(df.copy()) # Use a copy to avoid modifying the original df's categorical column

        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=150, bbox_inches='tight')
        plt.close(fig) # Important to close the figure to free up memory
        img_buffer.seek(0)
        yield img_buffer.getvalue()


    # --- Markdown to DataFrame Conversion ---
    def markdown_to_csv(llm_output):
        # Find all Markdown tables
        # Updated regex to be slightly more robust, looking for lines starting with |
        #tables = re.findall(r"(\n\|(?:[^\n]+\|)+\n\|(?:\s*-+\s*\|)+\n(?:\|(?:[^\n]+\|)+\n)+)", "\n" + llm_output + "\n")
        tables = re.findall(r"(\|(?:[^\n]+\|)+\n\|(?:\s*-+\s*\|)+\n(?:\|(?:[^\n]+\|)+\n)+)", llm_output)
        
        dataframes = []

        if tables:
            for table_string in tables:
                try:
                    # Split lines and extract columns
                    lines = table_string.strip().split("\n")
                    # Ensure there are at least 3 lines (header, separator, data)
                    #if len(lines) < 3:
                    #    print(f"Skipping malformed table (not enough lines): {table_string[:100]}...")
                    #    continue
                    if len(lines) >= 3:
                        headers = [col.strip() for col in lines[0].split("|")[1:-1]]  
                        data_rows = [line.split("|")[1:-1] for line in lines[2:]]  

                    #headers = [col.strip() for col in lines[0].split("|")[1:-1]]  
                    #if not headers or any(h == '' for h in headers):
                    #     print(f"Skipping malformed table (bad headers): {table_string[:100]}...")
                    #     continue

                    #data_rows = [line.split("|")[1:-1] for line in lines[2:]]  

                    #cleaned_data = []
                    #for row in data_rows:
                    #    if len(row) == len(headers):
                    #        cleaned_data.append([col.strip() for col in row])
                    #    else:
                    #        print(f"Skipping malformed data row (column count mismatch): {row}")

                    cleaned_data = []
                    if headers and not any(h == '' for h in headers):
                        for row in data_rows:
                            if len(row) == len(headers):
                                # Strip whitespace from each cell
                                cleaned_data.append([col.strip() for col in row])
                        else:
                            print(f"Skipping malformed data row (column count mismatch): {row}")
                        df = pd.DataFrame(cleaned_data, columns=headers)
                        dataframes.append(df)


                    #if cleaned_data:
                    #    df = pd.DataFrame(cleaned_data, columns=headers)
                    #    dataframes.append(df)
                    else:
                        print(f"No valid data rows found for table: {table_string[:100]}...")

                except Exception as e:
                    print(f"Error processing a potential Markdown table: {e}\nContent snippet: {table_string[:200]}...")
                    # Continue to the next potential table
                    continue

        return dataframes
    # --- End Markdown to DataFrame Conversion ---

    # --- Table Merging Function ---
    def merge_extracted_tables_llm(tables, parse_description):
        if not tables:
            return None # No tables to merge
        
        if not _ensure_llm_model():
            return pd.DataFrame({"Error": ["Modelo LLM no disponible para merge."]})
        model = llm_model_instance.get()

        # Convert DataFrames to Markdown strings
        table_strings = [table.to_markdown(index=False) for table in tables]

        # Create a prompt for the LLM (using the template from the Streamlit code)
        merge_template = ("You are tasked with merging the following Markdown tables into a single, comprehensive Markdown table.\n"
            "The tables contain information related to: {parse_description}.\n" # Using {parse_description} placeholder
            "Please follow these instructions carefully:\n\n"
            "1. **Task:** Merge the data from the following tables into a single table that matches the description: {parse_description}.\n" # Using {parse_description} placeholder
            "2. **Output Format:** Return the merged data ONLY as a single Markdown table. The table MUST be correctly formatted.\n"
            "3. **Markdown Table Format:** The table must adhere to the following Markdown format:\n"
            "   - Start with a header row, clearly labeling each column, separated by pipes (|).\n"
            "   - Follow the header row with an alignment row, using hyphens (-) to indicate column alignment (e.g., --- for left alignment).\n"
            "   - Subsequent rows should contain the data, with cells aligned according to the alignment row.\n"
            "   - Use pipes (|) to separate columns in each data row.\n"
            "4. **No Explanations:** Do not include any introductory or explanatory text before or after the table.\n"
            "5. **Empty Response:** If no information matches the description, return an empty string ('') if no data can be merged.\n"
            "6. **Duplicate Columns:** If there are duplicate columns, rename them to be unique.\n"
            "7. **Missing Values:** If there are missing values, fill them with 'N/A'.\n\n"
            "Here are the tables:\n\n" + "\n\n".join(table_strings) +
            "\n\nReturn the merged table in Markdown format:"
        )
        merge_prompt = ChatPromptTemplate.from_template(merge_template)

        try:
            # Invoke the LLM
            response = model.invoke(merge_prompt.format_prompt(parse_description=parse_description)) # Pass description to template
            merged_markdown = response.content
            
            # Convert the merged markdown back to a DataFrame
            merged_dfs = markdown_to_csv(merged_markdown)
            if merged_dfs:
                return merged_dfs[0] # Return the first (and hopefully only) merged table
            else:
                # If LLM output didn't produce a valid table, return the raw text
                print("LLM merge output did not produce a valid table.")
                return pd.DataFrame({"Mensaje": [f"El LLM intentó fusionar las tablas, pero el resultado no fue una tabla válida. Output del LLM:\n{merged_markdown}"]})

        except Exception as e:
            print(f"Error during LLM merge: {e}")
            return pd.DataFrame({"Error": [f"Error al fusionar tablas con LLM: {str(e)}"]})
    # --- End Table Merging Function ---

    # --- Scraper/Parser Output Display ---
    @output
    @render.ui
    def scraper_parser_output():
        results = scraper_parser_results.get()
        if results is None:
            return ui.markdown("Para este caso no se necesita usar ningún menú de los que se encuentran en la izquierda\n Presiona 'Scrapear y Parsear' para comenzar.")
        
        if "error" in results:
            return ui.markdown(f"**Error:** {results['error']}")

        raw_outputs = results.get("raw_outputs", [])
        extracted_tables = results.get("extracted_tables", [])
        merged_table = results.get("merged_table")

        output_elements = []
        ### Scenario 1 
        if merged_table is not None and not (isinstance(merged_table, pd.DataFrame) and ('Error' in merged_table.columns or 'Mensaje' in merged_table.columns)):
            print('Escenario 1: Tabla fusionada')
            output_elements.append(ui.tags.h5("Tabla Fusionada:", style="color: #6c757d;")) # Added some style
            output_elements.append(
                ui.HTML(merged_table.to_html(
                    classes="table table-dark table-striped table-hover table-sm table-bordered", 
                    escape=False, border=0 # border=0 from pandas, table-bordered from Bootstrap
                ))
            )

        ### Scenario 2 
        # Scenario 2: No valid merged table, but individual tables were extracted.
        # This happens if merging wasn't successful or if there was only one table to begin with (no merge needed).
        elif extracted_tables: # extracted_tables is a list of DFs from markdown_to_csv
            output_elements.append(ui.tags.h5("Tablas Extraídas Individualmente:", style="color: #6c757d;"))
            print('Escenario 2: Tabla extraída individualmente')
            # If merge was attempted and resulted in a message (e.g., LLM couldn't merge), show that message.
            if merged_table is not None and isinstance(merged_table, pd.DataFrame) and \
               ('Error' in merged_table.columns or 'Mensaje' in merged_table.columns):
                output_elements.append(ui.markdown(f"**Nota sobre la Fusión:** {merged_table.iloc[0].iloc[0]}"))

            for i, table_df in enumerate(extracted_tables):
                 # Ensure table_df is a DataFrame and not empty before rendering
                if isinstance(table_df, pd.DataFrame) and not table_df.empty:
                    output_elements.append(ui.tags.h6(f"Tabla Extraída {i+1}:", style="color: #adb5bd;"))
                    output_elements.append(
                        ui.HTML(table_df.to_html(
                            classes="table table-dark table-striped table-hover table-sm table-bordered", 
                            escape=False, border=0
                        ))
                    )
                 # This case should ideally not be hit if extract_tables_from_llm_outputs filters correctly
                 # and markdown_to_csv only returns valid DFs or an empty list.
                elif isinstance(table_df, pd.DataFrame) and ('Error' in table_df.columns or 'Mensaje' in table_df.columns):
                    output_elements.append(ui.markdown(f"**Información Tabla {i+1}:** {table_df.iloc[0].iloc[0]}"))
                 # else: # This case implies table_df is not a valid DataFrame to render
                     # output_elements.append(ui.markdown(f"**Tabla {i+1}:** No se pudo mostrar (formato inesperado)."))
        
        # Scenario 3: No tables (neither merged nor individual) were successfully extracted.
        # Display the raw output(s) from the LLM.
        elif raw_outputs: # raw_outputs is a list of strings (raw responses from LLM)
            output_elements.append(ui.tags.h5("Output del LLM (No se detectaron tablas formateadas):"))
            print('Escenario 3: Texto solo')

            # If merge was attempted (e.g., on an empty list of tables) and resulted in a message.
            if merged_table is not None and isinstance(merged_table, pd.DataFrame) and \
               ('Error' in merged_table.columns or 'Mensaje' in merged_table.columns):
                output_elements.append(ui.markdown(f"**Nota sobre la Fusión:** {merged_table.iloc[0].iloc[0]}"))

            # Concatenate and display all non-empty raw outputs
            # Each item in raw_outputs corresponds to a chunk processed by the LLM
            formatted_raw_outputs = []
            for i, output_text in enumerate(raw_outputs):
                if output_text and output_text.strip(): # Check if the string is not None and not just whitespace
                    formatted_raw_outputs.append(f"**Respuesta del LLM para el contenido/chunk {i+1}:**\n\n```text\n{output_text}\n```")
            
            if formatted_raw_outputs:
                output_elements.append(ui.markdown("\n\n---\n\n".join(formatted_raw_outputs)))
            else:
                output_elements.append(ui.markdown("_El LLM no devolvió contenido textual o solo espacios en blanco._"))
        else:
            output_elements.append(ui.markdown("El proceso se completó, pero no se generaron resultados (ni tablas ni texto crudo). Por favor, verifica la URL y la descripción del parseo."))

        return ui.tags.div(*output_elements)
    # --- End Scraper/Parser Output Display ---

    #@output
    #@render_widget
    #def emotion_output():
    #    #platform = input.platform_selector()
    #    #if platform == "wikipedia":
    #    #    df = process_wikipedia_for_df()
    #    #elif platform == "twitter":
    #    #    df = getTweetsResponses()
    #    #elif platform == "youtube":
    #    #    df = get_youtube_comments()
    #    #elif platform == "maps":
    #    #    df = mapsComments()
    #    #else:
    #    #    return None
    #    df = processed_dataframe.get()
    #    if df.empty or 'sentiment' not in df.columns:
    #        fig = px.bar(title='Análisis de Sentimiento (Sin datos)')
    #        fig.update_layout(xaxis_title="Sentimiento", yaxis_title="Conteo")
    #        return fig
    #    if 'Error' in df.columns:
    #        fig = px.bar(title=f"Análisis de Sentimiento (Error: {df['Error'].iloc[0]})")
    #        return fig
    #    return plot_emotion_distribution(df)

    @output
    @render.plot
    def emotion_plot_output():
        df = processed_dataframe.get()
        if not isinstance(df, pd.DataFrame) or df.empty or 'emotion' not in df.columns:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(10,7))
            ax.text(0.5, 0.5, 'Análisis de Emociones (Sin datos)', 
                    ha='center', va='center', fontsize=14, color='white')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.patch.set_facecolor('#222222')
            ax.set_facecolor('#222222')
            plt.tight_layout()
            return fig
        if 'Error' in df.columns and len(df) == 1: 
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(10,7))
            error_message = df['Error'].iloc[0]
            ax.text(0.5, 0.5, f"Análisis de Emociones\n(Error: {error_message})", 
                    ha='center', va='center', fontsize=14, color='white', wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.patch.set_facecolor('#222222')
            ax.set_facecolor('#222222')
            plt.tight_layout()
            return fig
        return plot_emotion_distribution_seaborn(df.copy())

    @render.download(filename='analisis_emociones.png')
    async def download_emotion_plot():
        df = processed_dataframe.get()
        if not isinstance(df, pd.DataFrame) or df.empty or 'emotion' not in df.columns:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(6,4))
            ax.text(0.5, 0.5, 'Análisis de Emociones\n(Sin datos para graficar)', ha='center', va='center', fontsize=12, color='white')
            ax.set_xticks([]); ax.set_yticks([])
            fig.patch.set_facecolor('#222222'); ax.set_facecolor('#222222')
        elif 'Error' in df.columns and len(df) == 1:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(6,4))
            error_message = df['Error'].iloc[0]
            ax.text(0.5, 0.5, f"Análisis de Emociones\n(Error: {error_message})", ha='center', va='center', fontsize=12, color='white', wrap=True)
            ax.set_xticks([]); ax.set_yticks([])
            fig.patch.set_facecolor('#222222'); ax.set_facecolor('#222222')
        else:
            fig = plot_emotion_distribution_seaborn(df.copy())

        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        img_buffer.seek(0)
        yield img_buffer.getvalue()

    # '''Topics Distribution'''
    @output
    @render.plot
    def topics_plot_output():
        df = processed_dataframe.get()
        if not isinstance(df, pd.DataFrame) or df.empty or 'topics' not in df.columns:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(10,7))
            ax.text(0.5, 0.5, 'Análisis de Tópicos (Sin datos)', 
                    ha='center', va='center', fontsize=14, color='white')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.patch.set_facecolor('#222222')
            ax.set_facecolor('#222222')
            plt.tight_layout()
            return fig
        if 'Error' in df.columns and len(df) == 1: 
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(10,7))
            error_message = df['Error'].iloc[0]
            ax.text(0.5, 0.5, f"Análisis de Tópicos\n(Error: {error_message})", 
                    ha='center', va='center', fontsize=14, color='white', wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.patch.set_facecolor('#222222')
            ax.set_facecolor('#222222')
            plt.tight_layout()
            return fig
        return plot_topics_distribution_seaborn(df.copy())

    @render.download(filename='analisis_topicos.png')
    async def download_topics_plot():
        df = processed_dataframe.get()
        if not isinstance(df, pd.DataFrame) or df.empty or 'topics' not in df.columns:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(6,4))
            ax.text(0.5, 0.5, 'Análisis de Tópicos\n(Sin datos para graficar)', ha='center', va='center', fontsize=12, color='white')
            ax.set_xticks([]); ax.set_yticks([])
            fig.patch.set_facecolor('#222222'); ax.set_facecolor('#222222')
        elif 'Error' in df.columns and len(df) == 1:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(6,4))
            error_message = df['Error'].iloc[0]
            ax.text(0.5, 0.5, f"Análisis de Tópicos\n(Error: {error_message})", ha='center', va='center', fontsize=12, color='white', wrap=True)
            ax.set_xticks([]); ax.set_yticks([])
            fig.patch.set_facecolor('#222222'); ax.set_facecolor('#222222')
        else:
            fig = plot_topics_distribution_seaborn(df.copy())

        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        img_buffer.seek(0)
        yield img_buffer.getvalue()


    @output 
    @render.ui
    def google_map_embed():
        map_data = map_coordinates.get() # map_data will contain {'place_id': ..., 'lat': ..., 'lng': ...}
        platform = input.platform_selector()
        api_key = MAPS_API_KEY

        if platform == "maps" and map_data:
            place_id = map_data.get('place_id')
            lat = map_data.get('lat')
            lng = map_data.get('lng')

            iframe_url = None
            if place_id:
                # Using place_id is generally better for markers and info cards
                iframe_url = f"https://www.google.com/maps/embed/v1/place?key={api_key}&q=place_id:{place_id}"
                # Optionally, provide center and zoom as hints, though 'place' mode often auto-adjusts
                # if lat and lng: # Usually not needed for 'place' mode as it centers on the place_id
                #     iframe_url += f"&center={lat},{lng}&zoom=15"
            elif lat and lng:
                # Fallback to view mode if only coordinates are available
                iframe_url = f"https://www.google.com/maps/embed/v1/view?key={api_key}&center={lat},{lng}&zoom=15"
            
            if iframe_url:
                return ui.HTML(f'<iframe src="{iframe_url}" width="100%" height="500px" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>')
            else:
                return ui.p("Datos de mapa incompletos para mostrar. Asegúrate de que el lugar sea válido.")
        elif platform == "maps" and not map_data: # This means map_coordinates.get() returned None
            return ui.p("No se pudieron obtener las coordenadas o el ID del lugar para mostrar el mapa. Asegúrate de que el lugar sea válido.")
        return ui.p("Selecciona 'Google Maps' en el panel izquierdo y realiza una búsqueda para ver el mapa aquí.")
    
    # ''' Chat de Gemini'''
    def _ensure_gemini_model():
        if gemini_model_instance.get() is None:
            if GEMINI_API_KEY:
                try:
                    print('Iniciando el modelo de Gemini')
                    genai.configure(api_key=GEMINI_API_KEY)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    gemini_model_instance.set(model)
                    print('Modelo de Gemini inició correctamente')
                except Exception as e:
                    print(f"Error al cargar el modelo de Gemini: {e}")
                    current_gemini_response.set(f'Error: al iniciar el modelo {e} ')
            else: 
                current_gemini_response.set("Error: Clave de Gemini no configurada")
                return False 
        return gemini_model_instance.get() is not None 

    def query_pinecone_for_context(query_text: str, top_k: int = 3) -> str: 
        if not _ensure_pinecone_client_and_index() or not _ensure_gemini_embeddings_model():
            print('No se pueden obtener la query de Pinecone: Cliente, índice o modelo de embedding no está listo')
            
            return ""
        
        try:
            print(f"Insertando la query a Pinecone: '{query_text[:10]}...'")
            query_embedding_list = embed_texts_gemini([query_text], task_type="RETRIEVAL_QUERY")
            if not query_embedding_list or not query_embedding_list[0]:
                print("Falló al insertar la query a Pinecone")
                return ""
            query_embedding = query_embedding_list[0]

            print(f"Llamando a Pinecone con el top_k={top_k}...")
            query_results = pinecone_index_instance.get().query(vector=query_embedding, top_k=top_k, include_metadata=True)
            
            context_parts = [match['metadata']['text'] for match in query_results['matches'] if 'metadata' in match and 'text' in match['metadata']]
            return "\n\n---\n\n".join(context_parts)
        except Exception as e:
            print(f"Error al hacer la query a Pinecone: {e}")
            return ""        

    @output
    @render.text    
    def gemini_response():
        #print('Entró en la función de respuesta de Gemini')
        #genai.configure(api_key=GEMINI_API_KEY)
        #model = genai.GenerativeModel('gemini-1.5-pro')
        #prompt = input.gemini_prompt()
        #if not prompt or not GEMINI_API_KEY:
        #    return "Error: Pregunta no válida o clave API no configurada."
        #
        #try: 
        #    response = model.generate_content(prompt)
        #    return response.text
        #except Exception as e:
        #    return f"Error al obtener respuesta de Gemini: {e}"   
        return current_gemini_response.get()
    
    
    @reactive.Effect
    @reactive.event(input.ask_gemini)
    def ask_gemini_handler():
       # Ejemplo de cómo usar el usuario autenticado:
        current_user = usuario_autenticado.get()
        if not current_user:
            current_gemini_response.set("Error: Debes iniciar sesión para usar esta función.")
            return        
        #print('Entró en la función de respuesta de Gemini')
        #genai.configure(api_key=GEMINI_API_KEY)
        #model = genai.GenerativeModel('gemini-1.5-pro')
        #prompt = input.gemini_prompt()
        user_prompt =input.gemini_prompt()
        #if not prompt or not GEMINI_API_KEY:
        #if not prompt:
        if not user_prompt: 
            current_gemini_response.set("Por favor escribe una pregunta")
            return
        if not _ensure_gemini_model():
            return 

        model = gemini_model_instance.get()
        #print("Enviando pregunta a Gemini")
        # Prepare context from processed_dataframe
        pinecone_context = ""
        if PINECONE_API_KEY and PINECONE_INDEX_NAME: 
            print("Attempting to retrieve context from Pinecone...")
            pinecone_context = query_pinecone_for_context(user_prompt, top_k=3)
            if pinecone_context:
                print(f"Retrieved context from Pinecone:\n{pinecone_context[:200]}...")

        data_context = ""
        df_for_context = processed_dataframe.get()
        
        if isinstance(df_for_context, pd.DataFrame) and not df_for_context.empty:
            if not ('Error' in df_for_context.columns and len(df_for_context) == 1) and \
               not ('Mensaje' in df_for_context.columns and len(df_for_context) == 1):
                # Only use data if it's not an error/message placeholder
                collapsed_text_for_context = collapse_text(df_for_context)
                if collapsed_text_for_context and isinstance(collapsed_text_for_context, str) and not collapsed_text_for_context.startswith("Error"):
                    # Limit context size if necessary, e.g., first 2000 characters
                    max_context_len = 4000 
                    if len(collapsed_text_for_context) > max_context_len:
                        data_context = f"Basado en los siguientes datos (extracto):\n---\n{collapsed_text_for_context[:max_context_len]}...\n---\n"
                    else:
                        data_context = f"Basado en los siguientes datos:\n---\n{collapsed_text_for_context}\n---\n"

        #final_prompt_to_gemini = f"{data_context}Pregunta del usuario: {user_prompt}"
        combined_context = ""
        if pinecone_context:
            combined_context += f"Contexto relevante de la base de conocimiento:\n{pinecone_context}\n\n---\n\n"
        final_prompt_to_gemini = f"{combined_context}{data_context}Pregunta del usuario: {user_prompt}"
        #print(f"Enviando pregunta a Gemini (con contexto si existe):\n{final_prompt_to_gemini[:50]}...") 
        print(f"Usuario '{current_user}' enviando pregunta a Gemini (con contexto si existe):\n{final_prompt_to_gemini[:500]}...") # Log más largo

        try:
            with ui.Progress(min=1, max=3) as p:
                #p.set(message="Generando respuesta...")
                p.set(message="Generando respuesta de Gemini...", detail="Contactando al modelo...")
                response = model.generate_content(final_prompt_to_gemini)
                #response = model.generate_content(prompt)
                #output.gemini_response.set(response.text)
                current_gemini_response.set(response.text)
                print("Respuesta de Gemini recibida")
        except Exception as e:
            #print('Entró en la función de respuesta de Gemini')
            #output.gemini_response.set(f"Error: {str(e)}")
            error_msg = f'Error al obtener respuesta de Gemini: {str(e)}'
            print(error_msg)
            current_gemini_response.set(error_msg)

    #@session.download(filename='datos_exportados.csv')
    @reactive.Effect
    def _generate_and_set_mind_map():
        mind_map_json_str = generate_mind_map_data_llm()
        
        if not mind_map_json_str or not isinstance(mind_map_json_str, str) or mind_map_json_str.startswith("Error:") or mind_map_json_str.startswith("No hay datos"):
            mind_map_html.set(f"<p><i>{mind_map_json_str}</i></p>")
            return

        try:
            # Limpiar el string JSON si Gemini añade ```json ... ```
            if mind_map_json_str.startswith("```json"):
                mind_map_json_str = mind_map_json_str.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(mind_map_json_str)
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])

            if not nodes:
                mind_map_html.set("<p><i>No se pudieron extraer nodos para el mapa mental.</i></p>")
                return

            #net = Network(notebook=True, height="750px", width="100%", cdn_resources='remote', directed=True)
            net = Network(notebook=True, height="750px", width="100%", cdn_resources='in_line', directed=True)
            
            for node in nodes:
                net.add_node(node["id"], label=node["label"], title=node["label"])
            
            for edge in edges:
                net.add_edge(edge["from"], edge["to"])

            # Guardar en un archivo temporal HTML y leer su contenido
            # Opciones de Pyvis para mejorar la visualización
            net.set_options("""
            var options = {
              "nodes": {
                "font": { "size": 12, "face": "Tahoma" }
              },
              "edges": {
                "arrows": {"to": { "enabled": true, "scaleFactor": 0.5 }},
                "smooth": { "type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.4 }
              },
              "layout": { "hierarchical": { "enabled": true, "sortMethod": "directed", "shakeTowards": "roots" } }
            }
            """)
            html_content = net.generate_html()
            mind_map_html.set(html_content)
        except Exception as e:
            mind_map_html.set(f"<p><i>Error al generar o visualizar el mapa mental: {str(e)}<br>Respuesta JSON recibida:<br><pre>{mind_map_json_str}</pre></i></p>")

    @output
    @render.ui
    def mind_map_output():
        html_content = mind_map_html.get()
        if html_content:
            return ui.HTML(html_content)
        return ui.p("Generando mapa mental... o esperando datos.")
    
    @render.download(filename='datos_exportados.csv')
    async def download_data():
        if input.platform_selector()=="scraper_parser_tab_internal":
            results = scraper_parser_results.get()
            if results and "merged_table" in results and isinstance(results["merged_table"], pd.DataFrame):
                yield results["merged_table"].to_csv(index=False, encoding='utf-8-sig')
                return 
        df_to_download = processed_dataframe.get()
        if isinstance(df_to_download, pd.DataFrame) and not df_to_download.empty:
            if 'Error' in df_to_download.columns and len(df_to_download)==1:
                yield df_to_download.to_csv(index=False, encoding='utf-8-sig')
            elif 'Mensaje' in df_to_download.columns and len(df_to_download) == 1:
                yield df_to_download.to_csv(index=False, encoding='utf-8-sig')
            else:
                yield df_to_download.to_csv(index=False, encoding='utf-8-sig')
        else:
            yield "No hay datos para descargar"

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
