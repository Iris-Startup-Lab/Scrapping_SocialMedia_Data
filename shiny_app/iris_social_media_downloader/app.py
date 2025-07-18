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

##### Importando librerías generales y principales
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
from datetime import datetime, timezone
#from shared import app_dir, tips
#### Librerías para análisis de datos y llamado de datos
import numpy as np 
import pandas as pd
#### Librerías para gráficos
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget, render_plotly 
from plotly.subplots import make_subplots
from PIL import Image 
from pyvis.network import Network
import base64
import io

#### Librerías para web scraping 
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium_stealth import stealth 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException

from google_play_scraper import app as play_app, reviews as play_reviews, Sort, reviews_all, search
import html


#### Librerías para uso de las API's oficiales
import googlemaps
import tweepy
from googleapiclient.discovery import build ## Youtube

### Librería de análisis de texto
import spacy

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
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine

#### Librerías para exportar a excel

import  openpyxl
import xlsxwriter
### Librería para generar UUID para la sesión
import uuid


import random 
from datetime import datetime, timedelta
import time 

### LLamando a los scripts de ayuda// Calling helper scripts
from config import * ### Script con secretos y también ligas a Chrome Driver etc 
from utils import *
from scrapers.webpage_scrapers import _maps_comments, _get_youtube_channels_and_comments, _get_tweets_from_twitter_api
#from large_and_small_models.gemini_model_and_functions import _ensure_gemini_model, topics_generator, current_gemini_response, gemini_embeddings_model, gemini_model_instance
from sentiments_and_emotions_classifier.emotions_and_sentiments import  spacy_nlp_sentiment, pysentimiento_analyzer_instance, pysentimiento_emotions_analyzer_instance, summarizer_pipeline_instance, emotion_model, emotion_tokenizer, emotion_config
from sentiments_and_emotions_classifier.emotions_and_sentiments import _ensure_spacy_sentiment_model,  _ensure_pysentimiento_analyzer, _ensure_pysentimient_emotions_analyzer,  generate_emotions_analysis, sentiment_based_on_emotions_analysis, generate_sentiment_analysis

################## Mensajes generales y funciones de ayuda ###########
random.seed(42)  

print(f"PostgreSQL Host configured: {'Yes' if PG_HOST else 'No'}")
print(f"PostgreSQL User configured: {'Yes' if PG_USER else 'No'}")


# --- End Helper functions ---




#### Comenzando la UI/Frontend
app_ui = ui.page_fixed(
    ui.tags.head(
        #ui.tags.link(rel="stylesheet", href="./www/styles.css"), # El archivo css de estulos
        #ui.tags.link(rel="stylesheet", href=(Path(__file__).parent / "www/styles.css") ), # El archivo css de estulos

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
                //});
            }); 
            // Nuevo para el handling de login para Enter en inicio de sesión           
            $('#email_login, #password_login').on('keypress', function(e){
                if(e.which == 13){
                    e.preventDefault();
                    $('#boton_login').click();
                }
            });
            // Asegurando que deban escribir contraseña para poder dar enter
            $('#password_login').on('keypress', function(e){
                if(e.which == 13){
                    e.preventDefault();
                    $('#boton_login').click();
                }
            });
        });

        """)
    ),
    #### """ui antiguo"""
    ui.output_ui("ui_app_dinamica"),
    #theme=shinyswatch.theme.darkly()
    #theme=shinyswatch.theme.minty()
    theme=shinyswatch.theme.zephyr()

)

#### Comenzando el server/Backend
def server(input, output, session):
    # Variables reactivas para la autenticación
    usuario_autenticado = reactive.Value(None)
    mensaje_login = reactive.Value("")
    #same_network_comparison_data_r = reactive.Value(None)
    # Configuración inicial de las variables para el server
    ## Lazy Load o Carga del Perezoso
    pinecone_client = reactive.Value(None)
    pinecone_index_instance = reactive.Value(None)
    processed_dataframe = reactive.Value(pd.DataFrame())
    current_gemini_response = reactive.Value("Carga datos y luego haz una pregunta sobre ellos, o haz una pregunta general. Presiona Enter o el botón verde a la izquierda para activar el bot")
    gemini_embeddings_model = reactive.Value(None)
    gemini_model_instance= reactive.Value(None)

    scraper_parser_results = reactive.Value(None)
    llm_model_instance = reactive.Value(None)
    topic_pipeline_instance =  reactive.Value(None)
    map_coordinates = reactive.Value(None)
    mind_map_html = reactive.Value(None)
    options = ClientOptions().replace(schema="iris_scraper") # If you need to modify default options
    client = create_client(SUPABASE_URL, SUPABASE_KEY, options=options)
    supabase_client_instance = reactive.Value(None)
    psycopg2_conn_instance = reactive.Value(None)
    active_comparison_session_id = reactive.Value(None)
    # Reactives para el módulo de comparación
    comparison_data_r = reactive.Value(None) # Almacenará el dict de DFs
    comparison_summary_r = reactive.Value("")
    comparison_chat_history_r = reactive.Value([]) # Para el chatbot comparativo
    infographic_status_r = reactive.Value("")
 
     # Reactive Values for Same-Network Comparison Module
    same_network_comparison_data_r = reactive.Value(None)
    same_network_comparison_summary_r = reactive.Value("")
    same_network_comparison_chat_history_r = reactive.Value([])
    same_network_infographic_status_r = reactive.Value("")
    #supabase_client_instance.set(client)

    # --- Comenzando la lógica para autenticación
    def ui_login_form():
        """Retorna la UI para el formulario de login."""
        return ui.div(
            ui.row(
                ui.column(4,
                          ui.panel_well(   # Añadimos una clase para targeting específico en CSS
                              ui.hr(), # Estilos de hr se manejarán en CSS
                              ui.h1("Iris Startup Lab", style="text-align: center; font-size: 2em;"), 
                              ui.h1("Presenta:", style="text-align: center; font-size: 0.8em;"), 
                              ui.h1('"ChismesitoGPT"', style="text-align: center; font-size: 1.5em; font-weight: bold;"), 
                              ui.h1("Obten datos de la web y redes sociales + IA", style="text-align: center; font-size: 0.8em;"), 
                              #ui.output_image("icon"),
                              ui.div(
                                  #ui.img(src=str(Path(__file__).parent / "www/Icon_chismoso_Gemini.png"), style="height: 60px; display: block; margin-left: auto; margin-right: auto; margin-top: 10px; margin-bottom: 10px;"), # Ajusta la ruta y el estilo
                                  ui.output_image("icon"),
                                  style="text-align: center;", class_="login-icon-container" 
                              ),
                              ui.h3("Acceso:", style="text-align: center; font-size: 1.2em;"), 
                              ui.hr(), # Estilos de hr se manejarán en CSS
                              ui.input_text("email_login", "Correo Electrónico:", placeholder="tucorreoelectronico@elektra/dialogus"),
                              ui.input_password("password_login", "Contraseña (No. Empleado):", placeholder="Tu número de empleado"),                              
                              #ui.input_action_button("boton_login", "Ingresar", class_="btn-primary btn-block"),
                              ui.input_action_button("boton_login", "Ingresar", class_="btn-primary btn-block mt-2"), # mt-2 para un poco de margen
                              ui.output_text("texto_mensaje_login"),
                              ui.hr(), 
                              ui.div( # Contenedor principal para los logos, centrado
                                  ui.div(ui.output_image("logo1"), style="display: inline-block; vertical-align: middle; margin-right: 10px;"),
                                  #ui.div(ui.output_image("logo2"), style="display: inline-block; vertical-align: middle;"),
                                  #ui.div(ui.output_image("logo1")),
                                  #ui.div(ui.output_image("logo2")),
                                  style="text-align: center;", class_="login-logo-container" 
                              ),                              
                              style="color: #00968b; margin-top: 10px; text-align: center;"
                              
                          ),
                        offset=4,
                        class_="login-panel-column" # Clase para la columna si es necesario
                ) # Fin ui.column
            ),
            style="margin-top: 100px;"
        )

    @render.image
    def logo1():
        from shiny.types import ImgData
        dir = Path(__file__).resolve().parent
        #img: ImgData = {"src": str(dir / "www/LogoNuevo.png"), "width": "100px; height: 50px; display: inline-block; margin-right: 1px;"}
        img: ImgData = {
            #"src": str(dir / "www/LogoNuevo.png"), 
            "src": str(dir / "www/Logos_GS_Iris.png"),
            "alt": "Logo 1",
            "style": "height: 70px; width: auto;" # Más pequeño, display se maneja en el div contenedor
        }        
        return img 
    
    @render.image
    def logo2():
        from shiny.types import ImgData
        dir = Path(__file__).resolve().parent
        #img: ImgData = {"src": str(dir / "www/GSNuevo.png"), "width": "100px; height: 50px; display: inline-block; margin-right: 1px;"}
        img: ImgData = {
            "src": str(dir / "www/GSNuevo.png"), 
            "alt": "Logo 2",
            "style": "height: 35px; width: auto;" # Más pequeño, display se maneja en el div contenedor
        }        
        return img 

    @render.image
    def icon():
        from shiny.types import ImgData
        dir = Path(__file__).resolve().parent
        img: ImgData = {
            "src": str(dir / "www/Icon_chismoso_Gemini2.png"), 
            "alt": "Icon",
            "style": "height: 300px; width: auto; display: block; margin-left: auto; margin-right: auto;" # Un poco más pequeño, display block para centrarlo solo
        }        
        return img 


    @output
    @render.text
    def current_session_id_display():
        session_id = active_comparison_session_id.get()
        if session_id:
            return f"ID Sesión Comparación: {session_id[:8]}..." # Muestra una parte
        return "No hay sesión de comparación activa."

    # Efecto para limpiar la sesión de comparación
    @reactive.Effect
    @reactive.event(input.clear_comparison_session)
    def _clear_session():
        active_comparison_session_id.set(None)
        ui.notification_show("Sesión de comparación limpiada.", duration=3)    
    

    @output
    @render.text
    def texto_mensaje_login():
        return mensaje_login.get()

    @reactive.Effect
    @reactive.event(input.boton_login)
    def manejar_intento_login():
        email = input.email_login()
        password_str = input.password_login() 

        if not email:
            mensaje_login.set("Por favor, ingrese su correo electrónico.")
            return
        
        if not password_str:
            mensaje_login.set("Por favor, ingrese su contraseña.")
            return
        
        if not _ensure_psycopg2_connection():
            mensaje_login.set("Error de conexión a Postgresql")
            return 
        #conn = psycopg2.conn_instance.get()
        conn  = psycopg2_conn_instance.get()
        cursor = None 
        try:
            cursor = conn.cursor()
            # La tabla es iris_scraper.iris_email_employees_enabled
            # Columnas: email (text), name (text), no_employee (bigint)
            sql_query = "SELECT name, no_employee FROM iris_scraper.iris_email_employees_enabled WHERE email = %s"
            cursor.execute(sql_query, (email.strip().lower(),))
            result = cursor.fetchone()

            if result:
                db_name, db_no_employee = result # db_no_employee es bigint
                
                # Comparamos la contraseña ingresada (string) con el no_employee de la BD (convertido a string)
                if str(db_no_employee) == password_str:
                    usuario_autenticado.set(db_name if db_name else email.split('@')[0]) # Usar el nombre de la BD o el prefijo del email
                    mensaje_login.set("")
                    ui.notification_show(f"¡Bienvenido, {usuario_autenticado.get()}!", type="message", duration=5)
                else:
                    mensaje_login.set("Contraseña incorrecta.")
                    usuario_autenticado.set(None)
            else:
                mensaje_login.set("Correo electrónico no encontrado o no autorizado.")
                usuario_autenticado.set(None)
        except psycopg2.Error as db_err:
            print(f"Error de base de datos durante el login: {db_err}")
            mensaje_login.set("Error al verificar credenciales. Intente más tarde.")
            usuario_autenticado.set(None)
        except Exception as e:
            print(f"Error inesperado durante el login: {e}")
            mensaje_login.set(f"Error inesperado durante el login: {e}")
            usuario_autenticado.set(None)
        finally:
            if cursor:
                cursor.close()
            # No cerramos la conexión global aquí, se reutiliza.
            # Si la conexión falla, _ensure_psycopg2_connection la reseteará en el próximo intento.


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


    # --- UI Principal de la Aplicación-
    @reactive.Calc
    def ui_principal_app():
        """Retorna la UI principal de la aplicación cuando el usuario está autenticado."""
        return ui.layout_sidebar(
            ui.sidebar(
                ui.output_ui("sidebar_dinamico"), # Contenido del sidebar cambiará
                width=350
            ),
            ui.navset_card_tab( # Pestañas principales
                nav_panel_comparacion_misma_red(),
                nav_panel_base_datos_y_chatbot(),
                nav_panel_analisis_y_visualizaciones(),
                #nav_panel_comparison_module(), # Nueva pestaña de comparación
                nav_panel_cross_source_comparison(),
                #nav_panel_modulo_comparacion(), 
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
                ui.input_text_area("parser_description", "Describe qué información quieres extraer: (Solo tablas de sitios web)", 
                                 placeholder="Ej: 'Tabla de precios de productos'", height=100, value = 'Genera una tabla con los precios de las motos de mayor a menor precio'),
                ui.input_action_button("execute_scraper_parser", "Scrapear y Parsear", 
                                       icon=ui.tags.i(class_="fas fa-play"), class_="btn-primary"),
                ui.hr(),
                ui.input_action_button("boton_logout", "Cerrar Sesión", class_="btn-danger btn-sm btn-block")
            )
        elif pestana_actual == "Módulo de Comparación misma red":
            return ui.div(
                    ui.markdown("### Comparación de Datos por Sesión"),
                    ui.output_text("active_session_id_for_comparison_display"),
                    ui.input_select("comparison_platform_selector", "Seleccionar Plataforma:",
                        {"twitter": "Twitter (X)", "youtube": "YouTube", "maps": "Google Maps", "reddit": "Reddit",
                        "facebook": "Facebook (Próximamente)", "tiktok": "TikTok (Próximamente)"}),
                    ui.input_text("comparison_query_1", "Marca/Tópico 1:", placeholder="Ej: 'iPhone 15' o '#ElonMusk'"),
                    ui.input_text("comparison_query_2", "Marca/Tópico 2:", placeholder="Ej: 'Samsung S24' o '#Tesla'"),
                    ui.input_text("comparison_query_3", "Marca/Tópico 3 (Opcional):", placeholder="Ej: 'Google Pixel'"),
                    #ui.input_text("comparison_query_4", "Marca/Tópico 4 (Opcional):", placeholder="Ej: 'OnePlus'"),
                    #ui.input_text("comparison_query_5", "Marca/Tópico 5 (Opcional):", placeholder="Ej: 'Xiaomi'"),
                    ui.hr()
            )
        else: # Para "Base de Datos y Chatbot" y "Análisis y Visualizaciones"
            return ui.div(
                ui.markdown(f"Usuario: **{usuario_autenticado.get()}**"),
                ui.markdown("**Social Media Downloader** - Extrae y analiza datos de diferentes plataformas."),
                ui.hr(),
                ui.input_select("platform_selector", "Seleccionar Plataforma:",
                    ##{"wikipedia": "Wikipedia", "playstore": "Google Play Store", "youtube": "YouTube", "maps": "Google Maps", "reddit": "Reddit", "twitter": "Twitter (X)", "generic_webpage": "Página web Genérica", "facebook": "Facebook (Próximamente)", "instagram": "Instagram (Próximamente)", "amazon_reviews": "Amazon Reviews (Próximamente)"}),
                    {"playstore": "Google Play Store", "youtube": "YouTube", "maps": "Google Maps", "reddit": "Reddit", "twitter": "Twitter (X)", "generic_webpage": "Página web Genérica", "wikipedia": "Wikipedia", "facebook": "Facebook (Próximamente)", "instagram": "Instagram (Próximamente)", "amazon_reviews": "Amazon Reviews (Próximamente)"}),

                ui.output_ui("platform_inputs"),
                ui.hr(),
                ui.input_checkbox('use_comparison_session', 'Iniciar/Continuar sesión de comparación', False ),
                ui.output_text("current_session_id_display"), # Para mostrar el ID actual
                ui.input_action_button("clear_comparison_session", "Limpiar Sesión de Comparación Actual", class_="btn-warning btn-sm mt-1"),
                ui.hr(),
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
                             ui.download_button("download_data_excel", "Descargar Excel", icon=ui.tags.i(class_="fas fa-file-excel"), class_="btn-success btn-sm mb-2 mt-2 ms-2"), # Botón Excel
                             icon=ui.tags.i(class_="fas fa-table-list")),
                ui.nav_panel("Resumen General", 
                             ui.output_ui('styled_summary_output'), 
                             #icon=ui.tags.i(class_="fas fa-file-lines")),
                             icon = ui.tags.i(class_ ="fa-solid fa-newspaper")),
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
                ui.nav_panel("Mapa (Solo al seleccionar Google Maps)", 
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
    
    def nav_panel_comparison_module():
        return ui.nav_panel(
            "Módulo de Comparación",
            ui.markdown("### Comparación de Datos por Sesión"),
            ui.output_text("active_session_id_for_comparison_display"),
            ui.input_action_button("execute_comparison", "Generar Comparación de la Sesión Actual", class_="btn-info mt-2 mb-3"),
            ui.hr(),
            #ui.output_ui("comparison_summary_output"),
            #ui.hr(),
            output_widget("comparison_sentiment_plot"),
            ui.hr(),
            output_widget("comparison_emotion_plot"),
            #ui.hr(),
            #output_widget("comparison_topics_plot"),
            icon=ui.tags.i(class_="fas fa-balance-scale")
        )
    
    #def nav_panel_modulo_comparacion():
    def nav_panel_cross_source_comparison():
        return ui.nav_panel(
            "Módulo de Comparación entre redes sociales",
            ui.navset_card_pill(
                ui.nav_panel("Análisis General Comparativo",
                   # ui.markdown("### Comparación de Datos por Sesión"),
                   # ui.output_text("active_session_id_for_comparison_display"),
                    ui.markdown("### Comparación de Datos en la Misma Red"),
                    ui.output_text("same_network_active_session_id_display"),
                    ui.input_action_button("execute_cross_source_comparison", "Generar/Actualizar Comparación de la Sesión Actual", class_="btn-info mt-2 mb-3"),
                    ui.hr(),
                    ui.output_ui("comparison_summary_output"),
                    ui.hr(),
                    output_widget("comparison_sentiment_plot"),
                    ui.hr(),
                    output_widget("comparison_emotion_plot"),
                    ui.hr(),
                    output_widget("comparison_topics_plot"),
                    icon=ui.tags.i(class_="fas fa-chart-bar")
                ),
                ui.nav_panel("Chatbot Comparativo entre fuentes",
                    ui.markdown("### Chatea con los Datos de la Sesión de Comparación"),
                    #ui.output_text("active_session_id_for_comparison_chat_display"),
                    ui.output_text("active_session_id_for_cross_source_comparison_chat_display"), # NEW: Separate chat display for cross-source
                    ui.hr(),
                    ui.input_text_area("comparison_chat_prompt", "Tu pregunta sobre los datos comparados:", rows=3, placeholder="Ej: ¿Cuáles son las principales diferencias entre las fuentes?"),
                    ui.input_action_button("ask_comparison_chat", "Preguntar al Chatbot Comparativo", class_="btn-primary mt-2"),
                    ui.hr(),
                    ui.card(
                        ui.card_header("Respuesta del Chatbot Comparativo"),
                        ui.output_ui("comparison_chat_response_output"), # Usamos output_ui para renderizar Markdown
                        style="min-height: 300px; overflow-y: auto;"
                    ),
                    icon=ui.tags.i(class_="fas fa-comments")
                ),
                ui.nav_panel("Generador de Infografía (Beta)",
                    ui.markdown("### Generar Infografía de la Comparación"),
                    ui.input_text("infographic_title", "Título para la Infografía:", placeholder="Ej: Comparativa de Opiniones sobre Producto X"),
                    #ui.input_action_button("generate_infographic_pdf", "Generar y Descargar Infografía (PDF)", class_="btn-success mt-2 mb-3"),
                    #ui.download_button("generate_infographic_pdf_handler", "Generar y Descargar Infografía (PDF)", class_="btn-success mt-2 mb-3"),                    
                    ui.download_button("generate_cross_source_infographic_pdf_handler", "Generar y Descargar Infografía (PDF)", class_="btn-success mt-2 mb-3"), # NEW: Separate infographic handler for cross-source
                    ui.output_ui("infographic_generation_status"), # Para mostrar mensajes de estado
                    icon=ui.tags.i(class_="fas fa-file-pdf")
                )
            ),
            icon=ui.tags.i(class_="fas fa-balance-scale")
        )

    def nav_panel_comparacion_misma_red():
        return ui.nav_panel(
            "Módulo de Comparación misma red",
            ui.navset_card_pill(
                ui.nav_panel("Análisis General Comparativo",
                    ui.output_ui("same_network_comparison_summary_output"), # NEW: Output for same-network summary
                    ui.input_action_button("execute_same_network_comparison", "Generar Comparación", class_="btn-info mt-2 mb-3"),
                    ui.hr(),
                    output_widget("same_network_sentiment_plot"),
                    ui.hr(),
                    output_widget("same_network_emotion_plot"),
                    icon=ui.tags.i(class_="fas fa-chart-bar")
                ),
                ui.nav_panel("Chatbot Comparativo Misma Red",
                    ui.markdown("### Chatea con los Datos de la Sesión de Comparación"),
                    ui.output_text("active_session_id_for_same_network_comparison_chat_display"),
                    ui.hr(),
                    ui.input_text_area("same_network_comparison_chat_prompt", "Tu pregunta sobre los datos comparados:", rows=3, placeholder="Ej: ¿Cuáles son las principales diferencias entre las fuentes?"),
                    ui.input_action_button("ask_same_network_comparison_chat", "Preguntar al Chatbot Comparativo", class_="btn-primary mt-2"),
                    ui.hr(),
                    ui.card(
                        ui.card_header("Respuesta del Chatbot Comparativo"),
                        ui.output_ui("same_network_comparison_chat_response_output"),
                        style="min-height: 300px; overflow-y: auto;"
                    ),
                    icon=ui.tags.i(class_="fas fa-comments")
                )
            ),
            icon=ui.tags.i(class_="fa-solid fa-scale-unbalanced-flip")
        )


    def nav_panel_scraper_tablas_chatbot():
        return ui.nav_panel(
            "Scrapear Tablas con Chatbot",
            # El contenido principal es posicional
            ui.card(
                ui.card_header("Resultados del Scraper y Parser"),
                ui.download_button("download_scraper_parser_table", "Descargar Tabla CSV", icon=ui.tags.i(class_="fas fa-download"), class_="btn-success btn-sm mb-2 mt-2"),
                ui.download_button("download_scraper_parser_table_excel", "Descargar Tabla Excel", icon=ui.tags.i(class_="fas fa-file-excel"), class_="btn-success btn-sm mb-2 mt-2 ms-2"), # Botón Excel
                ui.output_ui("scraper_parser_output"),
                style="overflow-y: auto;"
            ),
            icon=ui.tags.i(class_="fa-solid fa-wand-magic-sparkles") # El argumento de palabra clave 'icon' va al final
        )

    def fetch_comparison_data(session_id):
        if not _ensure_psycopg2_connection() or not session_id:
            return None # O un diccionario de DataFrames vacíos

        conn = psycopg2_conn_instance.get()
        all_data = {}
        table_names = [ # Lista de tus tablas que pueden tener session_id
            "wikipedia_data", "youtube_comments_data", "maps_reviews_data", 
            "twitter_posts_data", "generic_webpage_data", "reddit_comments_data",
            "playstore_reviews_data" # , "llm_web_parser_data"
        ]

        try:
            with conn.cursor() as cursor:
                for table in table_names:
                    query = f"SELECT * FROM iris_scraper.{table} WHERE session_id = %s"
                    cursor.execute(query, (session_id,))
                    results = cursor.fetchall()
                    if results:
                        colnames = [desc[0] for desc in cursor.description]
                        all_data[table] = pd.DataFrame(results, columns=colnames)
                    else:
                        all_data[table] = pd.DataFrame() # DataFrame vacío si no hay datos
            return all_data
        except Exception as e:
            print(f"Error fetching comparison data for session {session_id}: {e}")
            return None



    # --- Resto de tu lógica de servidor ---


    @render.image
    def app_logo():
        image_path = Path(__file__).parent / "www"/"LogoNuevo.png"
        return {"src": str(image_path), "alt": "App Logo"}

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
    
    
    # ----- Supabase ensure psql
    def _ensure_psycopg2_connection():
        if psycopg2_conn_instance.get() is None:
            if not all([PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD]):
                print("Error: PostgreSQL connection details not fully configured.")
                ui.notification_show("PostgreSQL no configurado. No se guardarán los datos.", type="error", duration=7)
                return False
            try:
                print("Attempting to connect to PostgreSQL...")
                conn = psycopg2.connect(
                    host=PG_HOST,
                    port=PG_PORT,
                    dbname=PG_DBNAME,
                    user=PG_USER,
                    password=PG_PASSWORD
                )
                psycopg2_conn_instance.set(conn)
                print("Successfully connected to PostgreSQL.")
                return True
            except Exception as e:
                print(f"Error connecting to PostgreSQL: {e}")
                ui.notification_show(f"Error al conectar con PostgreSQL: {e}", type="error", duration=7)
                return False
        return True    
        
    ### 
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


    def scrape_website(website_url, proxy=None):
        print(f"Attempting to scrape: {website_url}")
        
        unique_user_data_dir = None  
        driver = None               

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
            options.add_argument(f"user-agent={selected_user_agent}") 
            if proxy:
                options.add_argument(f'--proxy-server={proxy}') # Add proxy argument
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
                    # Apply stealth settings
                    try:
                        stealth(driver,
                                languages=["en-US", "en"],
                                vendor="Google Inc.",
                                platform="Win32",
                                webgl_vendor="Intel Inc.",
                                renderer="Intel Iris OpenGL Engine",
                                fix_hairline=True,
                                )
                        print("Selenium stealth applied.")
                    except Exception as stealth_e:
                        print(f"Error applying selenium-stealth: {stealth_e}")
                        # Continue without stealth if it fails
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # Scroll to bottom
                    time.sleep(random.uniform(1, 3))
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
                if proxy:
                    proxies = {'http': proxy, 'https': proxy}
                    page = requests.get(website_url, headers=agent, timeout=15, proxies=proxies)
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
            #if not _ensure_gemini_model(gemini_instance=gemini_model_instance, gemini_response=current_gemini_response):
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
        #if not _ensure_gemini_model(gemini_instance=gemini_model_instance, gemini_response=current_gemini_response):
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
    
    def save_df_to_postgres_with_psycopg2(df: pd.DataFrame, platform_name: str, requested_by: str, input_reference_value: str = None, session_id_value: str = None):
        if not _ensure_psycopg2_connection():
            return

        conn = psycopg2_conn_instance.get()
        
        table_name_map = {
            "wikipedia": "wikipedia_data",
            "youtube": "youtube_comments_data",
            "maps": "maps_reviews_data",
            "twitter": "twitter_posts_data",
            "generic_webpage": "generic_webpage_data",
            "reddit": "reddit_comments_data",
            "playstore": "playstore_reviews_data"
        }
        table_name = table_name_map.get(platform_name)

        if not table_name:
            ui.notification_show(f"Error: No hay tabla de PostgreSQL definida para la plataforma '{platform_name}'. Los datos no se guardaron.", type="error", duration=7)
            print(f"Error: No PostgreSQL table defined for platform '{platform_name}'.")
            return

        if df.empty or ('Error' in df.columns and len(df) == 1) or ('Mensaje' in df.columns and len(df) == 1):
            print(f"DataFrame para {platform_name} está vacío o es un error/mensaje. No se guarda en Supabase.")
            return

        # 1. Preparar df_insert para la INSERCIÓN de datos
        df_insert = df.copy()

        # Eliminar 'id' de df_insert si existe, ya que la BD lo generará.
        if "id" in df_insert.columns:
            df_insert = df_insert.drop(columns=["id"])
            print("Se eliminó la columna 'id' existente del DataFrame antes de la inserción.")

        # Añadir columnas de metadatos estándar
        df_insert["request_timestamp"] = datetime.now(timezone.utc) # Almacenar como datetime
        if input_reference_value is not None:
            df_insert["input_reference"] = input_reference_value
        if session_id_value is not None:
            df_insert["session_id"] = session_id_value
        df_insert["requested_by_user"] = requested_by

        # Replace NaT/NaN in the DataFrame that will be inserted
        df_insert = df_insert.replace({pd.NaT: None, np.nan: None})

        # Columnas para la sentencia INSERT
        quoted_insert_columns = [f'"{col_name}"' for col_name in df_insert.columns]
        cols_sql = ", ".join(quoted_insert_columns)
        
        qualified_table_name = f"iris_scraper.{table_name}"
        sql = f"INSERT INTO {qualified_table_name} ({cols_sql}) VALUES %s"
    
        data_tuples = [tuple(x) for x in df_insert.to_numpy()]

        # 2. Preparar df_for_schema_definition para la sentencia CREATE TABLE
        df_schema_base = df.copy()
        
        # Columnas que get_create_table_sql maneja explícitamente o que no son parte del esquema de datos
        cols_to_drop_for_schema = ["id"] # 'id' siempre es manejada por get_create_table_sql
        if input_reference_value is not None:
            # Si se provee input_reference_value, get_create_table_sql añadirá 'input_reference'.
            # Por lo tanto, si 'input_reference' existe en el df original, debe eliminarse de df_schema_base
            # para evitar definirla dos veces. Si no existe, eliminarla no causa daño.
            cols_to_drop_for_schema.append("input_reference")
        if session_id_value is not None: # Similarmente para session_id
            cols_to_drop_for_schema.append("session_id")

        df_schema_base = df_schema_base.drop(columns=cols_to_drop_for_schema, errors='ignore')
        # print(f"Columnas en df_schema_base después de eliminar id/input_reference: {df_schema_base.columns.tolist()}")

        # Añadir columnas de metadatos para la inferencia del esquema (sus tipos serán inferidos)
        df_for_schema_definition = df_schema_base.copy() # Empezar con la base limpia
        df_for_schema_definition["request_timestamp"] = pd.NaT # Para inferencia de dtype
        df_for_schema_definition["requested_by_user"] = ""   # Para inferencia de dtype
        # print(f"Columnas en df_for_schema_definition para get_create_table_sql: {df_for_schema_definition.columns.tolist()}")

        cursor = None
        try:
            cursor = conn.cursor()
            # Generate and execute CREATE TABLE IF NOT EXISTS
            #create_table_sql_stmt = get_create_table_sql(df_copy, qualified_table_name)
            #create_table_sql_stmt = get_create_table_sql(df_copy.drop(columns=["input_reference_internal_use"], errors='ignore'), qualified_table_name, has_input_reference_col=(input_reference_value is not None))
            # df_for_schema_definition does not contain 'input_reference' at this point.
            # get_create_table_sql will add 'input_reference TEXT' if (input_reference_value is not None) is True.
            create_table_sql_stmt = get_create_table_sql(
                df_for_schema_definition,
                qualified_table_name,
                has_input_reference_col=(input_reference_value is not None), 
                has_session_id_col=(session_id_value is not None)

            )            
            print(f"Ensuring table {qualified_table_name} exists...")
            # print(f"Schema SQL: {create_table_sql_stmt}") # For debugging the generated SQL
            cursor.execute(create_table_sql_stmt)
            # conn.commit() # DDL like CREATE TABLE IF NOT EXISTS is often auto-committed or can be committed here.
            execute_values(cursor, sql, data_tuples)
            conn.commit()
            ui.notification_show(f"{len(data_tuples)} registros guardados en PostgreSQL (tabla: '{qualified_table_name}').", type="message", duration=5)
            print(f"Se guardaron exitosamente {len(data_tuples)} registros en la tabla '{qualified_table_name}' de PostgreSQL.")
        except psycopg2.Error as e:
            if conn: 
                conn.rollback()
            ui.notification_show(f"Error de Psycopg2 al guardar en '{qualified_table_name}': {str(e)}", type="error", duration=7)
            print(f"Error de Psycopg2 al guardar en '{qualified_table_name}': {e}")           
            import traceback
            traceback.print_exc()
        except Exception as e: 
            if conn: 
                conn.rollback()
            ui.notification_show(f"Error general al guardar en '{qualified_table_name}': {str(e)}", type="error", duration=7)
            print(f"Error general al guardar en '{qualified_table_name}': {e}")        
        finally:
            if cursor:
                cursor.close()

    @reactive.Effect
    @reactive.event(input.execute)
    def handle_execute():
        platform = input.platform_selector()
        df = pd.DataFrame()
        map_coordinates.set(None)
        current_input_reference = None
        
        session_id_to_use = None # Default to no session ID
        if input.use_comparison_session(): # Check the state of the checkbox
            # Checkbox is ticked, so we want to use/create a session ID
            current_active_session = active_comparison_session_id.get()
            if current_active_session is None:
                # No active session ID yet, create a new one
                new_sid = str(uuid.uuid4())
                active_comparison_session_id.set(new_sid)
                session_id_to_use = new_sid
                ui.notification_show(f"Nueva sesión de comparación iniciada: {new_sid[:8]}...", type="info", duration=5)
            else:
                # An active session ID already exists, use it
                session_id_to_use = current_active_session
                ui.notification_show(f"Continuando sesión de comparación: {session_id_to_use[:8]}...", type="info", duration=3)
        # If input.use_comparison_session() is False, session_id_to_use remains None, and no session-related notification is shown here.
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
            
            if platform == "wikipedia": current_input_reference = input.wikipedia_url()
            elif platform == "youtube": current_input_reference = input.youtube_url()
            elif platform == "maps": current_input_reference = input.maps_query() # Corrected
            elif platform == "twitter": current_input_reference = input.twitter_query() # Corrected
            elif platform == "generic_webpage": current_input_reference = input.generic_webpage_url()
            elif platform == "reddit": current_input_reference = input.reddit_url()
            elif platform == "playstore": current_input_reference = input.playstore_url()
            
        processed_dataframe.set(df)
       # Guardar en Supabase si el df es válido y el usuario está logueado
        if isinstance(df, pd.DataFrame) and not df.empty and \
           not (('Error' in df.columns and len(df) == 1) or \
                ('Mensaje' in df.columns and len(df) == 1)):
            if usuario_autenticado.get(): # Solo guardar si el usuario está autenticado
                #save_df_to_supabase(df, platform, usuario_autenticado.get())
                save_df_to_postgres_with_psycopg2(df, platform, usuario_autenticado.get(), current_input_reference, session_id_value = session_id_to_use)
            else:
                # Esto solo se mostraría si se permite el scraping sin login
                #ui.notification_show("Usuario no autenticado. Datos extraídos pero no guardados en Supabase.", type="warning", duration=7)
                #print("Usuario no autenticado. Datos extraídos pero no guardados en Supabase.")
                ui.notification_show("Usuario no autenticado. Datos extraídos pero no guardados en PostgreSQL.", type="warning", duration=7)        
                print("Usuario no autenticado. Datos extraídos pero no guardados en PostgreSQL.")
        elif isinstance(df, pd.DataFrame) and df.empty:
            print(f"No se extrajeron datos para {platform}. Nada que guardar en Supabase.")
        else: # df contiene un error o mensaje
            print(f"La extracción para {platform} resultó en un error/mensaje. No se guarda en Supabase.")

    @reactive.Effect
    @reactive.event(input.execute_scraper_parser)
    def handle_scraper_parser_execute():
        urls_input = input.scraper_urls()
        parse_description = input.parser_description()
        current_user = usuario_autenticado.get()

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
        if not current_user:
            ui.notification_show("Usuario no autenticado. No se pueden guardar los datos parseados.", type="error", duration=5)
            scraper_parser_results.set({"error": "Usuario no autenticado. Los resultados no se guardarán."})
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
            """
            # Guardar merged_table_df en Supabase
            if isinstance(merged_table_df, pd.DataFrame) and not merged_table_df.empty and \
               not (('Error' in merged_table_df.columns and len(merged_table_df) == 1) or \
                    ('Mensaje' in merged_table_df.columns and len(merged_table_df) == 1)):
                if current_user: # Doble chequeo, aunque ya se hizo arriba
                    save_df_to_postgres_with_psycopg2(merged_table_df, "llm_web_parser", current_user)
            elif isinstance(merged_table_df, pd.DataFrame) and merged_table_df.empty:
                 print("El Parser LLM resultó en un DataFrame vacío. Nada que guardar en Supabase.")
            else: # merged_table_df contiene un error o mensaje
                 print("El Parser LLM resultó en un DataFrame de error/mensaje. No se guarda en Supabase.")
            """

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
            df['origin'] = "wikipedia"
            df['topics'] = df['text'].apply(classify_call)
        else: 
            df['text'] = df['text'].apply(lambda x: TranslateText(text=x, source= languageDetected, target='es'))
            df['sentiment'] = df['text'].apply(generate_sentiment_analysis)
            df['emotion'] = df['text'].apply(generate_emotions_analysis)
            classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
            df['topics'] = df['text'].apply(classify_call)
            df['origin'] = "wikipedia"

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
                df['origin'] = "generic_webpage"

            else: 
                df['text'] = df['text'].apply(lambda x: TranslateText(text=x, source= languageDetected, target='es'))
                df['sentiment'] = df['text'].apply(generate_sentiment_analysis)
                df['emotion'] = df['text'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['topics'] = df['text'].apply(classify_call)
                df['origin'] = "generic_webpage"
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
        max_results_count = 100
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
                df['origin'] = "twitter"

            else: 
                df['text'] = df['text'].apply(lambda x: TranslateText(text=x, source= languageDetected, target='es')) 
                df['sentiment'] = df['text'].apply(generate_sentiment_analysis)
                df['emotion'] = df['text'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['origin'] = "twitter"
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
                maxResults=100
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
                df['origin'] = "youtube"

            else: 
                df['comment'] = df['comment'].apply(lambda x: TranslateText(text=x, source= languageDetected, target='es')) 
                df['sentiment'] = df['comment'].apply(generate_sentiment_analysis)
                df['emotion'] = df['comment'].apply(generate_emotions_analysis)
                classify_call = partial(generate_zero_shot_classification_with_labels, candidate_labels=actual_candidate_labels)
                df['origin'] = "youtube"
                df['topics'] = df['comment'].apply(classify_call)

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
                df['origin'] = "maps"
                df['topics'] = df['comment'].apply(classify_call)
            else: 
                df['comment'] = df['comment'].apply(lambda x: TranslateText(text=x, source= languageDetected, target='es')) 
                df['sentiment'] = df['comment'].apply(generate_sentiment_analysis)
                df['emotion'] = df['comment'].apply(generate_emotions_analysis)
                df['origin'] = "maps"
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
                df['origin'] = "reddit"
                df['topics'] = df['comment'].apply(classify_call)
            
            else: 
                df['comment'] = df['comment'].apply(lambda x: TranslateText(text=x, source= languageDetected, target='es')) 
                df['sentiment'] = df['comment'].apply(generate_sentiment_analysis)
                df['emotion'] = df['comment'].apply(generate_emotions_analysis)
                df['origin'] = "reddit"
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
                    count=200, 
                    filter_score_with=None 
                )
            else:            
                app_id = re.search(r'(?<=id=)[^&]+', playstore_url).group(0)
                reviews = play_reviews(
                    app_id,
                    lang='es', 
                    country='mx', 
                    sort=Sort.NEWEST, 
                    count=200, 
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
                df['origin'] = "playstore"
                df['topics'] = df['content'].apply(classify_call)
            else: 
                df['content'] = df['content'].apply(lambda x: TranslateText(text=x, source= languageDetected, target='es')) 
                df['sentiment'] = df['content'].apply(generate_sentiment_analysis)
                df['emotion'] = df['content'].apply(generate_emotions_analysis)
                df['origin'] = "playstore"
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


    def generate_seaborn_count_plot_bytes(data_dict, data_key_for_df, x_column, hue_column, title, category_orders=None, color_palette=None):
        """
        Genera un gráfico de conteo con Seaborn y devuelve los bytes de la imagen PNG.
        
        Args:
            data_dict (dict): Diccionario que contiene DataFrames por origen.
            data_key_for_df (str): La clave principal en data_dict para obtener el DataFrame (ej. 'youtube_df').
                                O un DataFrame ya combinado si se prefiere.
            x_column (str): Nombre de la columna para el eje X.
            hue_column (str): Nombre de la columna para la segmentación por color (hue).
            title (str): Título del gráfico.
            category_orders (dict, optional): Diccionario para especificar el orden de las categorías.
                                            Ej: {'sentiment': ['Positivo', 'Neutral', 'Negativo']}
            color_palette (dict or list, optional): Paleta de colores para Seaborn.
        """
        
        # 1. Preparar los datos: Combinar DataFrames de data_dict si es necesa
        """
        all_dfs = []
        for origin, df_source in data_dict.items():
            if isinstance(df_source, pd.DataFrame) and not df_source.empty and \
            x_column in df_source.columns and hue_column in df_source.columns:
                # Seleccionamos solo las columnas necesarias para el plot y añadimos 'origin'
                # Si x_column es 'origin', entonces el hue_column es la categoría principal
                temp_df = df_source[[x_column, hue_column]].copy()
                # Si x_column no es 'origin', pero quieres graficar por 'origin', ajusta aquí.
                # Para este ejemplo, asumimos que x_column podría ser 'origin' y hue_column la categoría.
                if x_column != 'origin_column_name_in_df': # Reemplaza 'origin_column_name_in_df' si es diferente
                    temp_df['origin_plot_col'] = origin # Añade una columna de origen si x_column no lo es
                all_dfs.append(temp_df)

        if not all_dfs:
            print(f"Seaborn: No hay datos suficientes para graficar '{title}'.")
            return None
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        """
        combined_df = None
        if data_dict is None and isinstance(data_key_for_df, pd.DataFrame):
            # Case 1: data_key_for_df is the DataFrame to plot directly
            combined_df = data_key_for_df.copy() # Use a copy
            if not (x_column in combined_df.columns and hue_column in combined_df.columns):
                print(f"Seaborn: DataFrame provided to data_key_for_df is missing required columns ('{x_column}' or '{hue_column}') for '{title}'.")
                return None
        elif isinstance(data_dict, dict) and data_dict:
            # Case 2: data_dict is provided, combine DFs from it
            all_dfs_list = []
            for origin_name, df_source_item in data_dict.items():
                if isinstance(df_source_item, pd.DataFrame) and not df_source_item.empty and \
                   x_column in df_source_item.columns and hue_column in df_source_item.columns:
                    temp_df = df_source_item[[x_column, hue_column]].copy()
                    # If x_column is not 'origin' but you intend to group by origin from the dict keys:
                    if x_column == 'origin': # Assuming x_column is literally 'origin'
                        temp_df[x_column] = origin_name # Ensure the origin from dict key is used if x_column is 'origin'
                    all_dfs_list.append(temp_df)
            
            if not all_dfs_list:
                print(f"Seaborn: No valid DataFrames found in data_dict to combine for '{title}'.")
                return None
            combined_df = pd.concat(all_dfs_list, ignore_index=True)
        else:
            print(f"Seaborn: Invalid arguments or no data provided for '{title}'. data_dict type: {type(data_dict)}, data_key_for_df type: {type(data_key_for_df)}")
            return None

        if combined_df is None or combined_df.empty:
            print(f"Seaborn: No data available to plot for '{title}' after processing inputs.")
            return None
        
        # Determinar el orden de las categorías para 'hue' si se proporciona
        hue_order_list = None
        if category_orders and hue_column in category_orders:
            hue_order_list = category_orders[hue_column]

        # Crear el gráfico
        plt.figure(figsize=(10, 6)) # Ajusta el tamaño según necesites
        sns.countplot(
            data=combined_df,
            x=x_column, # Columna para el eje X (p.ej., 'origin_plot_col' si la creaste)
            hue=hue_column, # Columna para diferenciar por color (p.ej., 'sentiment')
            order=category_orders.get(x_column) if category_orders else None, # Orden para las barras del eje X
            hue_order=hue_order_list, # Orden para las categorías del hue
            palette=color_palette
        )
        plt.title(title, fontsize=15)
        plt.ylabel("Cantidad", fontsize=12)
        plt.xlabel(x_column.replace('_plot_col','').capitalize(), fontsize=12) # Etiqueta X más limpia
        plt.xticks(rotation=45, ha="right")
        #plt.legend(title=hue_column.capitalize())
        if hue_order_list or combined_df[hue_column].nunique() > 1 : # Show legend if hue is meaningful
            plt.legend(title=hue_column.capitalize(), loc='upper right')
        elif plt.gca().get_legend() is not None: # Remove legend if not needed (e.g. single hue category)
             plt.gca().get_legend().remove()        
        plt.tight_layout() # Ajusta el layout para que todo encaje bien

        # Guardar en un buffer de bytes
        img_bytes_io = io.BytesIO()
        plt.savefig(img_bytes_io, format='png', dpi=100) # dpi controla la resolución
        plt.close() # Importante cerrar la figura para liberar memoria

        img_bytes_io.seek(0) # Rebobinar el buffer al inicio
        return img_bytes_io.getvalue()



    
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
        #if not _ensure_gemini_model(gemini_instance=gemini_model_instance, gemini_response=current_gemini_response):
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
        extracted_tables_list = results.get("extracted_tables", []) # Renombrado para claridad
        merged_table_df = results.get("merged_table") # Renombrado para claridad

        output_elements = []
        table_displayed = False

        # Prioridad 1: Mostrar tabla fusionada si es válida
        if isinstance(merged_table_df, pd.DataFrame) and not merged_table_df.empty and not ('Error' in merged_table_df.columns or 'Mensaje' in merged_table_df.columns):
            print('Mostrando Tabla Fusionada')
            output_elements.append(ui.tags.h5("Tabla Fusionada:", style="color: #6c757d;"))
            output_elements.append(
                ui.HTML(merged_table_df.to_html(
                    classes="table table-dark table-striped table-hover table-sm table-bordered", 
                    escape=False, border=0 # border=0 from pandas, table-bordered from Bootstrap
                ))
            )
            table_displayed = True
        # Prioridad 2: Mostrar la primera tabla extraída válida si la fusionada no está disponible o no es válida
        elif not table_displayed and extracted_tables_list:
            first_valid_extracted_table = None
            for i, table_df in enumerate(extracted_tables_list):
                if isinstance(table_df, pd.DataFrame) and not table_df.empty and not ('Error' in table_df.columns or 'Mensaje' in table_df.columns):
                    first_valid_extracted_table = table_df
                    print(f'Mostrando Tabla Extraída Individualmente #{i+1}')
                    output_elements.append(ui.tags.h5(f"Tabla Extraída Individualmente #{i+1}:", style="color: #6c757d;"))
                    output_elements.append(
                        ui.HTML(table_df.to_html(
                            classes="table table-dark table-striped table-hover table-sm table-bordered", 
                            escape=False, border=0
                        ))
                    )
                    table_displayed = True
                    break # Solo mostrar la primera válida
            
            # Si después de iterar no se mostró ninguna tabla extraída válida,
            # y la tabla fusionada era un error/mensaje, mostrar ese mensaje.
            if not table_displayed and isinstance(merged_table_df, pd.DataFrame) and \
               ('Error' in merged_table_df.columns or 'Mensaje' in merged_table_df.columns):
                output_elements.append(ui.markdown(f"**Información sobre la Fusión:** {merged_table_df.iloc[0].iloc[0]}"))

        # Prioridad 3: Mostrar mensaje de error/info de la tabla fusionada si no se mostró ninguna tabla antes
        elif not table_displayed and isinstance(merged_table_df, pd.DataFrame) and \
             ('Error' in merged_table_df.columns or 'Mensaje' in merged_table_df.columns):
            print('Mostrando Mensaje/Error de Tabla Fusionada')
            output_elements.append(ui.markdown(f"**Información:** {merged_table_df.iloc[0].iloc[0]}"))
            table_displayed = True # Consideramos que se mostró "algo"

        # Prioridad 4: Mostrar raw_outputs si no se mostró ninguna tabla
        if not table_displayed and raw_outputs:
            output_elements.append(ui.tags.h5("Output del LLM (No se detectaron tablas formateadas para mostrar):", style="color: #6c757d;"))
            print('Mostrando Raw LLM Output')

            # Concatenate and display all non-empty raw outputs
            formatted_raw_outputs = []
            for i, output_text in enumerate(raw_outputs):
                if output_text and output_text.strip(): # Check if the string is not None and not just whitespace
                    formatted_raw_outputs.append(f"**Respuesta del LLM para el contenido/chunk {i+1}:**\n\n```text\n{output_text}\n```")
            
            if formatted_raw_outputs:
                output_elements.append(ui.markdown("\n\n---\n\n".join(formatted_raw_outputs)))
            else:
                output_elements.append(ui.markdown("_El LLM no devolvió contenido textual procesable o solo espacios en blanco._"))
        
        # Si después de todo no hay nada que mostrar
        if not output_elements:
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
        #if not _ensure_gemini_model(gemini_instance=gemini_model_instance, gemini_response=current_gemini_response):
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
            
            #print(f"DEBUG: Attempting to parse mind map JSON: '{mind_map_json_str}'")
            data = json.loads(mind_map_json_str)
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])

            if not nodes:
                mind_map_html.set("<p><i>No se pudieron extraer nodos para el mapa mental.</i></p>")
                # También es útil imprimir esto en la consola
                print("WARNING: No nodes found in parsed mind map JSON.")
                print(f"DEBUG: Parsed data was: {data}")
                return

            #net = Network(notebook=True, height="750px", width="100%", cdn_resources='remote', directed=True)
            net = Network(notebook=True, height="750px", width="100%", cdn_resources='in_line', directed=True)
            
            for node in nodes:
                net.add_node(node["id"], label=node["label"], title=node["label"])
            
            for edge in edges:
                net.add_edge(edge["from"], edge["to"])

            # Guardar en un archivo temporal HTML y leer su contenido
            # Opciones de Pyvis para mejorar la visualización
            #net.set_options("""
            #var options = {
            #  "nodes": {
            #    "font": { "size": 12, "face": "Tahoma" }
            #  },
            #  "edges": {
            #    "arrows": {"to": { "enabled": true, "scaleFactor": 0.5 }},
            #    "smooth": { "type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.4 }
            #  },
            #  "layout": { "hierarchical": { "enabled": true, "sortMethod": "directed", "shakeTowards": "roots" } }
            #}
            #""")


            # Dejamos el set_options original por si acaso, pero lo sobreescribiremos

            # Construir las opciones como un diccionario Python
            pyvis_options = {
                "nodes": {
                    "font": {"size": 14, "face": "Tahoma", "color": "#333333"},
                    "shape": "box",
                    "margin": 10,
                    "widthConstraint": {"maximum": 200}
                },
                "edges": {
                    "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}},
                    "color": {"color": "#848484", "highlight": "#848484", "hover": "#848484", "inherit": False},
                    "smooth": {"type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.4}
                },
                "layout": {
                    "hierarchical": {
                        "enabled": True,
                        "sortMethod": "directed",
                        "shakeTowards": "roots",
                        "direction": "UD",
                        "levelSeparation": 150,
                        "nodeSpacing": 100
                    }
                },
                "interaction": {
                    "dragNodes": True,
                    "dragView": True,
                    "zoomView": True,
                    "navigationButtons": True,
                    "tooltipDelay": 200
                },
                "physics": {
                    "enabled": False
                }
            }
            # Convertir el diccionario de Python a un string JSON
            options_json_string = json.dumps(pyvis_options)

            # Crear el string JavaScript para set_options
            # Es importante que options_json_string se inserte tal cual, json.dumps ya se encarga de las comillas.
            #js_options_string = f"var options = {options_json_string};"
            js_options_string = f"var options = {options_json_string}"

            net.set_options(js_options_string)


            html_content = net.generate_html()
            mind_map_html.set(html_content)
        except json.JSONDecodeError as je:
            print(f"ERROR: JSONDecodeError for mind map. Raw string was: '{mind_map_json_str}'")
            print(f"ERROR: JSONDecodeError details: {je}")
            # Mostrar el contexto del error si es posible
            context_window = 20
            error_pos = je.pos
            context_start = max(0, error_pos - context_window)
            context_end = min(len(je.doc), error_pos + context_window)
            error_context = je.doc[context_start:context_end]
            print(f"ERROR CONTEXT (around char {error_pos}): ...{error_context}...")
            mind_map_html.set(f"<p><i>Error de formato JSON al generar el mapa mental: {str(je)}<br>Contexto del error (alrededor del carácter {error_pos}):<br><pre>...{error_context}...</pre><br>Respuesta JSON completa recibida:<br><pre>{mind_map_json_str}</pre></i></p>")
        except Exception as e:
            print(f"ERROR: Failed to parse or visualize mind map JSON. Raw string was: '{mind_map_json_str}'")
            print(f"ERROR: Exception details: {e}")
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

    @render.download(filename ='datos_exportados.xlsx')
    async def download_data_excel():
        df_to_download = None 
        if input.platform_selector()=="scraper_parser_tab_internal":
            results = scraper_parser_results.get()
            if results and "merged_table" in results and isinstance(results["merged_table"], pd.DataFrame) and not results["merged_table"].empty:
                if not (('Error' in results["merged_table"].columns and len(results["merged_table"])==1) or \
                        ('Mensaje' in results["merged_table"].columns and len(results["merged_table"])==1)):
                    df_to_download = results["merged_table"]                
        if df_to_download is None:
            current_df = processed_dataframe.get()
            if isinstance(current_df, pd.DataFrame) and not current_df.empty:
                if not (('Error' in current_df.columns and len(current_df)==1) or \
                        ('Mensaje' in current_df.columns and len(current_df)==1)):
                    df_to_download = current_df
        if df_to_download is not None and isinstance(df_to_download, pd.DataFrame):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_to_download.to_excel(writer, index=False, sheet_name="datos")            
            yield output.getvalue()
        else: 
            yield "No hay datos válidos para descargar en formato Excel"


    # --- Lógica del Módulo de Comparación ---
    @output
    @render.text
    def active_session_id_for_comparison_display():
        session_id = active_comparison_session_id.get()
        if session_id:
            return f"Comparando datos para la Sesión ID: {session_id}"
        return "No hay sesión de comparación activa para mostrar. Inicia una desde la pestaña 'Base de Datos y Chatbot'."
    @output
    @render.text
    def same_network_active_session_id_display():
        platform = input.comparison_platform_selector()
        queries = [
            input.comparison_query_1(),
            input.comparison_query_2(),
            input.comparison_query_3(),
            input.comparison_query_4(),
            input.comparison_query_5(),
        ]
        queries = [q.strip() for q in queries if q.strip()]
        
        if platform and queries:
            return f"Comparando en {platform.upper()}: {', '.join(queries)}"
        elif platform:
            return f"Plataforma seleccionada: {platform.upper()}. Ingrese tópicos para comparar."
        return "Seleccione una plataforma e ingrese tópicos para comparar."


    @output
    @render.text
    def active_session_id_for_comparison_chat_display():
        session_id = active_comparison_session_id.get()
        if session_id:
            return f"Contexto del chat: Sesión ID {session_id}"
        return "No hay sesión de comparación activa para el chatbot."


    def fetch_and_analyze_comparison_data(platform: str, queries: TypingList[str]):
        all_queries_data = {}
        for i, query_text in enumerate(queries):
            if not query_text.strip():
                continue # Skip empty queries
            
            df_query = pd.DataFrame()
            query_label = f"Query {i+1}: {query_text}" # Label for this specific query

            try:
                if platform == "twitter":
                    df_query = _get_tweets_from_twitter_api(query_text)
                elif platform == "youtube":
                    df_query = _get_youtube_channels_and_comments(query_text, max_results_comments=5)
                elif platform == "maps":
                    df_query = _maps_comments(query_text)
                # Add other platforms as needed

                if not df_query.empty and not ('Error' in df_query.columns or 'Mensaje' in df_query.columns):
                    # Add query label and apply analysis
                    df_query['query_label'] = query_label
                    df_query['sentiment'] = df_query[df_query.columns.intersection(['text', 'comment', 'content'])[0]].apply(generate_sentiment_analysis)
                    df_query['emotion'] = df_query[df_query.columns.intersection(['text', 'comment', 'content'])[0]].apply(generate_emotions_analysis)
                    
                    # For topics, you might need a combined text from this specific query's DF
                    collapsed_query_text = collapse_text(df_query)
                    if collapsed_query_text:
                        # This part might need to be adapted if topics_generator expects a single string
                        # and generate_zero_shot_classification_with_labels expects candidate labels.
                        # For simplicity, let's assume a general topic classification for now.
                        # Or, if topics_generator returns a list of labels, use that.
                        # For now, let's just use a placeholder or skip if topics are too complex for this direct application.
                        df_query['topics'] = "No aplica en comparación" # Placeholder
                    
                    all_queries_data[query_label] = df_query
                else:
                    print(f"No data or error for query '{query_text}' on platform '{platform}'.")
            except Exception as e:
                print(f"Error fetching data for query '{query_text}' on platform '{platform}': {e}")
        return all_queries_data

   ## Funciones de sesión
    @reactive.Effect
    @reactive.event(input.execute_cross_source_comparison)
    async def _execute_comparison_logic():
        session_id = active_comparison_session_id.get()
        if not session_id:
            ui.notification_show("No hay una sesión de comparación activa.", type="warning", duration=5)
            # Limpia los reactivos de este módulo si no hay sesión
            comparison_data_r.set(None)
            comparison_chat_history_r.set([])
            infographic_status_r.set("")
            comparison_summary_r.set("Por favor, active una sesión de comparación y ejecute algunas búsquedas primero.")
            return
        ui.notification_show(f"Generando comparación para la sesión: {session_id[:8]}...", type="info", duration=4)
        # Obtiene todos los datos de diferentes tablas para el session_id actual
        fetched_data = fetch_comparison_data(session_id)
        comparison_data_r.set(fetched_data)

        if not fetched_data or all(df.empty for df in fetched_data.values()):
            ui.notification_show(f"No se encontraron datos para la sesión ID: {session_id}", type="warning", duration=5)
            comparison_chat_history_r.set([])
            infographic_status_r.set("")
            comparison_summary_r.set(f"No se encontraron datos para la sesión ID: {session_id}")
            return

        # Genera el resumen, especificando que NO es una comparación en la misma red
        summary = get_comparison_summary_gemini(fetched_data, is_same_network=False)
        comparison_summary_r.set(summary)
        
        # Prepara el historial del chatbot para este módulo
        ui.notification_show("Resumen comparativo entre fuentes generado.", type="success", duration=5)
        comparison_chat_history_r.set([
            {"role": "model", "parts": [{"text": f"Hola. Estoy listo para responder preguntas sobre los datos de la sesión de comparación (ID: {session_id[:8]}). El resumen general está en la pestaña 'Análisis General Comparativo'."}]}
        ])

    @reactive.Effect
    @reactive.event(input.execute_same_network_comparison)
    async def handle_same_network_comparison():
        """
        Orquesta la comparación de diferentes consultas dentro de la misma red social.
        1. Recopila las consultas y la plataforma desde la UI.
        2. Llama a `fetch_and_analyze_comparison_data_same_network` para scrapear y analizar los datos para cada consulta.
        3. Almacena los datos obtenidos en `same_network_comparison_data_r`.
        4. Genera un resumen con Gemini, indicando que es una comparación en la misma red (`is_same_network=True`).
        5. Almacena el resumen y prepara el historial del chatbot para este módulo específico.
        """
        queries = [
            input.comparison_query_1(),
            input.comparison_query_2(),
            input.comparison_query_3()#,
            #input.comparison_query_4(),
            #input.comparison_query_5(),
        ]
        queries = [q.strip() for q in queries if q.strip()]
        platform = input.comparison_platform_selector()

        if not queries or not platform:
            ui.notification_show("Por favor, seleccione una plataforma e ingrese al menos una marca/tópico para comparar.", type="warning", duration=5)
            same_network_comparison_data_r.set(None)
            same_network_comparison_summary_r.set("No hay datos para comparar. Ingrese los parámetros requeridos.")
            return

        ui.notification_show(f"Generando comparación para {platform} con las queries: {', '.join(queries)}...", type="info", duration=5)
        
        # Obtiene y analiza nuevos datos para las consultas especificadas
        all_queries_data = await fetch_and_analyze_comparison_data_same_network(platform, queries)
        same_network_comparison_data_r.set(all_queries_data)

        if not all_queries_data or all(df.empty for df in all_queries_data.values()):
            ui.notification_show(f"No se encontraron datos para las queries especificadas en {platform}.", type="warning", duration=5)
            same_network_comparison_chat_history_r.set([])
            same_network_infographic_status_r.set("")
            same_network_comparison_summary_r.set(f"No se encontraron datos para las queries especificadas en {platform}.")
            return

        # Genera el resumen, especificando que SÍ es una comparación en la misma red
        summary = get_comparison_summary_gemini(all_queries_data, is_same_network=True)
        same_network_comparison_summary_r.set(summary)
        
        # Prepara el historial del chatbot para este módulo
        ui.notification_show("Comparación en la misma red generada.", type="success", duration=5)
        same_network_comparison_chat_history_r.set([
            {"role": "model", "parts": [{"text": f"Hola, estoy listo para responder preguntas sobre la comparación de {platform} para las queries: {', '.join(queries)}."}]}
        ])


    """
    @reactive.Effect
    @reactive.event(input.execute_same_network_comparison)
    async def handle_same_network_comparison(): 
    #async def _execute_comparison_logic():
        session_id = active_comparison_session_id.get()
        if not session_id:
            return 
        #if not session_id:
        #    ui.notification_show("No hay una sesión de comparación activa.", type="warning", duration=5)
        #    comparison_data_r.set(None)
        #    comparison_chat_history_r.set([]) # Limpiar historial del chat si no hay sesión
        #    infographic_status_r.set("")            
        #    comparison_summary_r.set("Por favor, active una sesión de comparación y ejecute algunas búsquedas primero.")
        #    return

        ui.notification_show(f"Generando comparación para la sesión: {session_id[:8]}...", type="info", duration=3)
        
        # Simular trabajo asíncrono si es necesario para no bloquear la UI
        # await asyncio.sleep(0.1) 
        fetched_data = None
        #fetched_data = fetch_comparison_data(session_id)
        #comparison_data_r.set(fetched_data)

        #if not fetched_data or all(df.empty for df in fetched_data.values()):
        #    ui.notification_show(f"No se encontraron datos para la sesión ID: {session_id}", type="warning", duration=5)
        #    comparison_chat_history_r.set([])
        #    infographic_status_r.set("")
        #    comparison_summary_r.set(f"No se encontraron datos para la sesión ID: {session_id}")
        #    return
        queries = [
            input.comparison_query_1(),
            input.comparison_query_2(),
            input.comparison_query_3(),
            input.comparison_query_4(),
            input.comparison_query_5(),
        ]
        queries = [q.strip() for q in queries if q.strip()] 

        if not queries:
            ui.notification_show("Por favor, ingrese al menos una marca/tópico para comparar.", type="warning", duration=5)
            comparison_data_r.set(None)
            same_network_comparison_summary_r.set("No hay datos para comparar. Ingrese al menos una marca/tópico.")
            #comparison_summary_r.set("No hay datos para comparar. Ingrese al menos una marca/tópico.")
            return
        
        platform = input.comparison_platform_selector()
        if not platform:
            ui.notification_show("Por favor, seleccione una plataforma para comparar.", type="warning", duration=5)
            comparison_data_r.set(None)
            #comparison_summary_r.set("No hay datos para comparar. Seleccione una plataforma.")
            same_network_comparison_summary_r.set("No hay datos para comparar. Ingrese al menos una marca/tópico.")

            return          
        ui.notification_show(f"Generando comparación para {platform} con las queries: {', '.join(queries)}...", type="info", duration=5)          
        #all_queries_data = fetch_and_analyze_comparison_data(platform, queries)
        all_queries_data = await fetch_and_analyze_comparison_data_same_network(platform, queries)

        #comparison_data_r.set(all_queries_data)
        same_network_comparison_data_r.set(all_queries_data)
        if not all_queries_data or all(df.empty for df in all_queries_data.values()):
            ui.notification_show(f"No se encontraron datos para las queries especificadas en {platform}.", type="warning", duration=5)
            comparison_chat_history_r.set([])
            infographic_status_r.set("")
            #comparison_summary_r.set(f"No se encontraron datos para la sesión ID: {session_id}")
            same_network_comparison_summary_r.set(f"No se encontraron datos para las queries especificadas en {platform}.") # CHANGED: Use same_network summary
            return 
        # Generar resumen comparativo con Gemini
        #summary = get_comparison_summary_gemini(fetched_data)
        #summary = get_comparison_summary_gemini(all_queries_data)
        #summary = get_comparison_summary_gemini(fetched_data, is_same_network=False)
        summary = get_comparison_summary_gemini(all_queries_data, is_same_network=True)

        #summary = get_comparison_summary_gemini(all_queries_data, is_same_network=True)
        #comparison_summary_r.set(summary)
        same_network_comparison_summary_r.set(summary)
        #ui.notification_show("Resumen comparativo generado.", type="success", duration=4)
        #comparison_chat_history_r.set([{"role": "model", "parts": [{"text": f"Resumen inicial de la comparación (Sesión {session_id[:8]}):\n{summary}"}]}])
        #ui.notification_show("Resumen comparativo generado y disponible en 'Análisis General'.", type="success", duration=5)
        #comparison_chat_history_r.set([{"role": "model", "parts": [{"text": f"Hola. Estoy listo para responder preguntas sobre los datos de la sesión de comparación (ID: {session_id[:8]}). El resumen general está en la pestaña 'Análisis General Comparativo'."}]}])
        ui.notification_show("Comparación generada y resumen disponible.", type="success", duration=5)
        #comparison_chat_history_r.set([{"role": "model", "parts": [{"text": f"Hola. Estoy listo para responder preguntas sobre los datos de la comparación de {platform} para las queries: {', '.join(queries)}."}]}])
        same_network_comparison_chat_history_r.set([{"role": "model", "parts":[{"text": f"Hola, Estoy listo para responder preguntas sobre los datos de la comparación de {platform} para las queries: {', '.join(queries)}."}]}])
    """

    def get_comparison_summary_gemini(all_data_dict: Dict[str, pd.DataFrame], is_same_network: bool):
        #if not _ensure_gemini_model(gemini_instance=gemini_model_instance, gemini_response=current_gemini_response):
        if not _ensure_gemini_model():
            return "Error: Modelo de Gemini no disponible."

        gemini_model = gemini_model_instance.get()
        full_text_for_comparison = ""
        if is_same_network:
            task_description = "Has recopilado datos de la misma red social para diferentes marcas o tópicos. Tu tarea es comparar y contrastar las opiniones, sentimientos y temas clave entre estas marcas/tópicos. Los datos se agrupan por la marca/tópico."
            for query_label, df_source in all_data_dict.items():
                if not df_source.empty:
                    text_col = None
                    # Determine the text column dynamically
                    if 'text' in df_source.columns: text_col = 'text'
                    elif 'comment' in df_source.columns: text_col = 'comment'
                    elif 'content' in df_source.columns: text_col = 'content'
                    if text_col:
                        source_text = " ".join(df_source[text_col].dropna().astype(str).tolist())
                        if source_text.strip():
                            #full_text_for_comparison += f"\n\n--- Datos de {origin_value} ---\n{source_text[:2000]}..." # Limitar texto por fuente
                            full_text_for_comparison += f"\n\n--- Datos para '{query_label}' ---\n{source_text[:2000]}..." # Limit text per query
        else:
            task_description = "Has recopilado datos de varias fuentes sobre temas posiblemente relacionados. Tu tarea es generar un resumen sintetizado que compare y contraste las ideas principales, sentimientos predominantes y temas clave entre las diferentes fuentes. Los datos se agrupan por la fuente de origen."
            for _table_name, df_source in all_data_dict.items():
                if not df_source.empty and 'origin' in df_source.columns:
                    origin_value = df_source['origin'].iloc[0] if not df_source['origin'].empty else "Fuente Desconocida"
                    text_col = None
                    if 'text' in df_source.columns: text_col = 'text'
                    elif 'comment' in df_source.columns: text_col = 'comment'
                    elif 'content' in df_source.columns: text_col = 'content'

                    if text_col:
                        source_text = " ".join(df_source[text_col].dropna().astype(str).tolist())
                        if source_text.strip():
                            full_text_for_comparison += f"\n\n--- Datos de {origin_value} ---\n{source_text[:2000]}..." # Limitar texto por fuente


        if not full_text_for_comparison.strip():
            return "No hay texto combinado de las fuentes para resumir."

        max_len_prompt = 15000 
        if len(full_text_for_comparison) > max_len_prompt:
            full_text_for_comparison = full_text_for_comparison[:max_len_prompt] + "\n...(texto truncado)..."

        comparison_prompt = (
            f"Eres un analista experto. {task_description}\n\n"
            "A continuación, se presenta un conjunto de textos extraídos. "
            "Tu tarea es generar un resumen sintetizado que compare y contraste las ideas principales, sentimientos predominantes y temas clave entre las diferentes fuentes. "
            "Destaca similitudes, diferencias y cualquier patrón interesante que observes.\n\n"
            f"Textos combinados de las fuentes:\n{full_text_for_comparison}\n\n"
            "Resumen Comparativo Sintetizado:"
        )
        
        try:
            print(f"Enviando a Gemini para resumen comparativo: {comparison_prompt[:300]}...")
            response = gemini_model.generate_content(comparison_prompt)
            return response.text
        except Exception as e:
            return f"Error al generar resumen comparativo con Gemini: {e}"


    """
    #def get_comparison_summary_gemini(all_data_dict: Dict[str, pd.DataFrame]):
    def get_comparison_summary_gemini(all_data_dict: Dict[str, pd.DataFrame], is_same_network: bool):
        if not _ensure_gemini_model():
            return "Error: Modelo de Gemini no disponible."

        gemini_model = gemini_model_instance.get()
        full_text_for_comparison = ""
        if is_same_network:
            task_description = "Has recopilado datos de la misma red social para diferentes marcas o tópicos. Tu tarea es comparar y contrastar las opiniones, sentimientos y temas clave entre estas marcas/tópicos."
            task_description = "Has recopilado datos de la misma red social para diferentes marcas o tópicos. Tu tarea es comparar y contrastar las opiniones, sentimientos y temas clave entre estas marcas/tópicos. Los datos se agrupan por la marca/tópico."
        else:
            task_description = "Has recopilado datos de varias fuentes sobre temas posiblemente relacionados. Tu tarea es generar un resumen sintetizado que compare y contraste las ideas principales, sentimientos predominantes y temas clave entre las diferentes fuentes."
            task_description = "Has recopilado datos de varias fuentes sobre temas posiblemente relacionados. Tu tarea es generar un resumen sintetizado que compare y contraste las ideas principales, sentimientos predominantes y temas clave entre las diferentes fuentes. Los datos se agrupan por la fuente de origen."

        # Iterate through the dictionary where keys are query_labels and values are DataFrames
        for query_label, df_source in all_data_dict.items():
            if not df_source.empty:
                text_col = None
                # Determine the text column dynamically
                if 'text' in df_source.columns: text_col = 'text'
                elif 'comment' in df_source.columns: text_col = 'comment'
                elif 'content' in df_source.columns: text_col = 'content'
                if text_col:
                    source_text = " ".join(df_source[text_col].dropna().astype(str).tolist())
                    if source_text.strip():
                        #full_text_for_comparison += f"\n\n--- Datos de {origin_value} ---\n{source_text[:2000]}..." # Limitar texto por fuente
                        full_text_for_comparison += f"\n\n--- Datos para '{query_label}' ---\n{source_text[:2000]}..." # Limit text per query

        full_text_for_comparison = ""
        for _table_name, df_source in all_data_dict.items():
            if not df_source.empty and 'origin' in df_source.columns:
                origin_value = df_source['origin'].iloc[0] if not df_source['origin'].empty else "Fuente Desconocida"
                text_col = None
                if 'text' in df_source.columns: text_col = 'text'
                elif 'comment' in df_source.columns: text_col = 'comment'
                elif 'content' in df_source.columns: text_col = 'content'

                if text_col:
                    source_text = " ".join(df_source[text_col].dropna().astype(str).tolist())
                    if source_text.strip():
                        full_text_for_comparison += f"\n\n--- Datos de {origin_value} ---\n{source_text[:2000]}..." # Limitar texto por fuente
                        #full_text_for_comparison += f"\n\n--- Datos para '{query_label}' ---\n{source_text[:2000]}..." # Limit text per query


        if not full_text_for_comparison.strip():
            return "No hay texto combinado de las fuentes para resumir."

        max_len_prompt = 15000 
        if len(full_text_for_comparison) > max_len_prompt:
            full_text_for_comparison = full_text_for_comparison[:max_len_prompt] + "\n...(texto truncado)..."

        comparison_prompt = (
            f"Eres un analista experto. {task_description}\n"
            "A continuación, se presenta un conjunto de textos extraídos. "
            "Tu tarea es generar un resumen sintetizado que compare y contraste las ideas principales, sentimientos predominantes y temas clave entre las diferentes fuentes. "
            "Destaca similitudes, diferencias y cualquier patrón interesante que observes.\n\n"
            f"Textos combinados de las fuentes:\n{full_text_for_comparison}\n\n"
            "Resumen Comparativo Sintetizado:"
        )
        
        try:
            print(f"Enviando a Gemini para resumen comparativo: {comparison_prompt[:300]}...")
            response = gemini_model.generate_content(comparison_prompt)
            return response.text
        except Exception as e:
            return f"Error al generar resumen comparativo con Gemini: {e}"
    """

    @output
    @render.ui
    def comparison_summary_output():
        summary = comparison_summary_r.get()
        if not summary:
            return ui.p("Presiona 'Generar Comparación' para ver el resumen.")
        return ui.card(
            ui.card_header("Resumen Comparativo (Gemini)"),
            ui.markdown(summary)
        )
    @output
    @render.ui
    def same_network_comparison_summary_output(): # NEW: Separate summary output for same network
        summary = same_network_comparison_summary_r.get()
        if not summary: return ui.p("Presiona 'Generar Comparación' para ver el resumen.")
        return ui.card(
            ui.card_header("Resumen Comparativo Misma Red (Gemini)"),
            ui.markdown(summary)
        )

    def plot_comparison_stacked_bar(all_data_dict: Dict[str, pd.DataFrame], column_name: str, title: str, category_orders_dict: Dict = None, color_discrete_map_dict: Dict = None, x_axis_type: str = 'origin'):
    #def plot_comparison_stacked_bar(all_data_dict: Dict[str, pd.DataFrame], column_name: str, title: str, category_orders_dict: Dict = None, color_discrete_map_dict: Dict = None):
        plot_data_list = []
        if not all_data_dict: # Si all_data_dict es None o vacío
            all_data_dict = {}

        #for _table_name, df_source in all_data_dict.items():
        #for query_label, df_source in all_data_dict.items():
        for key_label, df_source in all_data_dict.items():
            if isinstance(df_source, pd.DataFrame) and not df_source.empty and column_name in df_source.columns and 'origin' in df_source.columns:
                counts = df_source[column_name].value_counts(dropna=False).reset_index()
                counts.columns = [column_name, 'count']
                if x_axis_type=='origin':
                    counts['x_axis_label'] = df_source['origin'].iloc[0] if 'origin' in df_source.columns and not df_source['origin'].empty else key_label
                elif x_axis_type=='query':
                    counts['x_axis_label'] = key_label
                else:
                    counts['x_axis_label'] = 'Desconocido'
                plot_data_list.append(counts)

        if not plot_data_list:
            fig = go.Figure()
            #fig.update_layout(title=f"{title} (Sin datos)", xaxis_title="Origen", yaxis_title="Porcentaje")
            fig.update_layout(title=f"{title} (Sin datos)", xaxis_title="Categoría", yaxis_title="Porcentaje")
            return fig

        combined_df = pd.concat(plot_data_list).fillna({column_name: "No especificado"}) # Rellenar NaNs en la columna de categoría
        
        #total_counts_per_origin = combined_df.groupby('origin')['count'].sum().reset_index(name='total_count')
        #combined_df = pd.merge(combined_df, total_counts_per_origin, on='origin', how='left')
        #combined_df['percentage'] = (combined_df['count'] / combined_df['total_count'].replace(0, 1)) * 100 # Evitar división por cero
        #total_counts_per_query = combined_df.groupby('query_label')['count'].sum().reset_index(name='total_query_count')
        total_counts_per_x_label = combined_df.groupby('x_axis_label')['count'].sum().reset_index(name='total_count')
        combined_df = pd.merge(combined_df, total_counts_per_x_label, on='x_axis_label', how='left')
        combined_df['percentage'] = (combined_df['count'] / combined_df['total_count'].replace(0, 1)) * 100        
        combined_df = combined_df.fillna(0)
        x_axis_title = "Plataforma de Origen" if x_axis_type == 'origin' else "Marca/Tópico"

        fig = px.bar(combined_df, x='x_axis_label', y='percentage', color=column_name,
                     title=title,
                     labels={'x_axis_label': x_axis_title, 'percentage': 'Porcentaje (%)', column_name: column_name.capitalize()},
                     category_orders=category_orders_dict if category_orders_dict else None,
                     color_discrete_map=color_discrete_map_dict if color_discrete_map_dict else None,
                     text='percentage')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
        fig.update_layout(yaxis_ticksuffix='%', barmode='stack', legend_title_text=column_name.capitalize())
        return fig

    @output
    @render_plotly
    def comparison_sentiment_plot():
        data_dict = comparison_data_r.get()
        sentiment_categories = ['Positivo', 'Neutral', 'Negativo', "Error: Análisis de sentimiento (Pysentimiento) fallido", "Error: Modelos de sentimiento no disponibles", "No especificado"]
        sentiment_colors = {
            'Positivo': '#2ECC71', 'Neutral': '#F1C40F', 'Negativo': '#E74C3C',
            "Error: Análisis de sentimiento (Pysentimiento) fallido": "#CCCCCC",
            "Error: Modelos de sentimiento no disponibles": "#AAAAAA",
            "No especificado": "#DDDDDD"
        }
        return plot_comparison_stacked_bar(data_dict, 'sentiment', 'Comparación de Sentimientos por Origen', 
                                           category_orders_dict={'sentiment': sentiment_categories},
                                           color_discrete_map_dict=sentiment_colors, x_axis_type='origin')

    @output
    @render_plotly
    def comparison_emotion_plot():
        data_dict = comparison_data_r.get()
        emotion_categories_es = ["Alegría", "Tristeza", "Enojo", "Miedo", "Sorpresa", "Asco", "Neutral", "Desconocida", "Error: Análisis de emociones fallido", "No especificado"]
        emotion_palette_es = {
            "Alegría": "#4CAF50", "Tristeza": "#2196F3", "Enojo": "#F44336", "Miedo": "#9C27B0",
            "Sorpresa": "#FFC107", "Asco": "#795548", "Neutral": "#9E9E9E", "Desconocida": "#607D8B",
            "Error: Análisis de emociones fallido": "#CCCCCC", "No especificado": "#DDDDDD"
        }
        #return plot_comparison_stacked_bar(data_dict, 'emotion', 'Comparación de Emociones por Origen',
        return plot_comparison_stacked_bar(data_dict, 'emotion', 'Comparación de Emociones por Marca/Tópico',
                                           category_orders_dict={'emotion': emotion_categories_es})

    
    
    """
    def plot_topics_comparison_grouped_bar(all_data_dict: Dict[str, pd.DataFrame], title: str, x_axis_type: str = 'origin'):   
        plot_data_list = []
        if not all_data_dict: all_data_dict = {}

        #for _table_name, df_source in all_data_dict.items():
        #for query_label, df_source in all_data_dict.items(): 
        #    if isinstance(df_source, pd.DataFrame) and not df_source.empty and 'topics' in df_source.columns and 'origin' in df_source.columns:
        #         origin_value = df_source['origin'].iloc[0] if not df_source['origin'].empty else "Desconocido"
        for key_label, df_source in all_data_dict.items():
                if isinstance(df_source, pd.DataFrame) and not df_source.empty and 'topics' in df_source.columns:
                    if x_axis_type == 'origin':
                        counts['x_axis_group'] = df_source['origin'].iloc[0] if not df_source['origin'].empty else "Desconocido"
                    elif x_axis_type == 'query':
                        counts['x_axis_group'] = key_label
                    else: 
                        counts['x_axis_group'] = 'Desconocido'
                    counts = df_source['topics'].value_counts(dropna=False).reset_index()
                    counts.columns = ['topic', 'count']
            #        counts['origin'] = origin_value
                    plot_data_list.append(counts)

        if not plot_data_list:
            fig = go.Figure()
            fig.update_layout(title=f"{title} (Sin datos)", xaxis_title="Tópico", yaxis_title="Conteo")
            return fig

        combined_df = pd.concat(plot_data_list).fillna({'topic': "No especificado"})
        
        # Limitar a los N tópicos más frecuentes en general para evitar gráficos muy cargados
        top_n_topics = 15
        overall_topic_counts = combined_df.groupby('topic')['count'].sum().nlargest(top_n_topics).index
        combined_df_filtered = combined_df[combined_df['topic'].isin(overall_topic_counts)]

        x_axis_title = "Plataforma de Origen" if x_axis_type=='origin' else 'Marca/Tópico'
        fig = px.bar(combined_df_filtered, x='topic', y='count', color='x_axis_group',
        #fig = px.bar(combined_df_filtered, x='topic', y='count', color='origin',
                     title=title,
                     labels={'topic': 'Tópico', 'count': 'Número de Menciones', 'x_axis_group': x_axis_title},
                     #labels={'topic': 'Tópico', 'count': 'Número de Menciones', 'origin': 'Plataforma de Origen'},
                     barmode='group')
        fig.update_xaxes(categoryorder='total descending', tickangle=-45)
        return fig
    """
    ### Funcion anterior para evaluar 
    def plot_topics_comparison_grouped_bar(all_data_dict: Dict[str, pd.DataFrame], title: str):
        plot_data_list = []
        if not all_data_dict: all_data_dict = {}

        for _table_name, df_source in all_data_dict.items():
            if isinstance(df_source, pd.DataFrame) and not df_source.empty and 'topics' in df_source.columns and 'origin' in df_source.columns:
                origin_value = df_source['origin'].iloc[0] if not df_source['origin'].empty else "Desconocido"
                counts = df_source['topics'].value_counts(dropna=False).reset_index()
                counts.columns = ['topic', 'count']
                counts['origin'] = origin_value
                plot_data_list.append(counts)

        if not plot_data_list:
            fig = go.Figure()
            fig.update_layout(title=f"{title} (Sin datos)", xaxis_title="Tópico", yaxis_title="Conteo")
            return fig

        combined_df = pd.concat(plot_data_list).fillna({'topic': "No especificado"})
        
        # Limitar a los N tópicos más frecuentes en general para evitar gráficos muy cargados
        top_n_topics = 15
        overall_topic_counts = combined_df.groupby('topic')['count'].sum().nlargest(top_n_topics).index
        combined_df_filtered = combined_df[combined_df['topic'].isin(overall_topic_counts)]

        fig = px.bar(combined_df_filtered, x='topic', y='count', color='origin',
                     title=title,
                     labels={'topic': 'Tópico', 'count': 'Número de Menciones', 'origin': 'Plataforma de Origen'},
                     barmode='group')
        fig.update_xaxes(categoryorder='total descending', tickangle=-45)
        return fig


    @output
    @render_plotly
    def comparison_topics_plot():
        data_dict = comparison_data_r.get()
        return plot_topics_comparison_grouped_bar(data_dict, 'Comparación de Tópicos por Origen')
        #return plot_topics_comparison_grouped_bar(data_dict, 'Comparación de Tópicos por Origen', x_axis_type='origin')

    # --- Inicio Lógica del Chatbot Comparativo ---
    ### AQUI
    # --- Lógica para el Chatbot Comparativo ---
    # --- Nuevos plots para la comparación en la misma red ---
    @output
    @render_plotly
    def same_network_sentiment_plot():
        data_dict = same_network_comparison_data_r.get() # CAMBIADO
        sentiment_categories = ['Positivo', 'Neutral', 'Negativo', "Error: Análisis de sentimiento (Pysentimiento) fallido", "Error: Modelos de sentimiento no disponibles", "No especificado"]
        sentiment_colors = {
            'Positivo': '#2ECC71', 'Neutral': '#F1C40F', 'Negativo': '#E74C3C',
            "Error: Análisis de sentimiento (Pysentimiento) fallido": "#CCCCCC",
            "Error: Modelos de sentimiento no disponibles": "#AAAAAA",
            "No especificado": "#DDDDDD"
        }
        #return plot_comparison_stacked_bar(data_dict, 'sentiment', 'Comparación de Sentimientos por Marca/Tópico',
        #return plot_comparison_stacked_bar(data_dict, 'emotion', 'Comparación de Emociones por Origen', # This is for cross-source comparison
        return plot_comparison_stacked_bar(data_dict, 'sentiment', 'Comparación de Sentimientos por Marca/Tópico',
                                           category_orders_dict={'sentiment': sentiment_categories},
                                           color_discrete_map_dict=sentiment_colors, x_axis_type='query')

    @output
    @render_plotly
    def same_network_emotion_plot():
        data_dict = same_network_comparison_data_r.get() # CAMBIADO
        emotion_categories_es = ["Alegría", "Tristeza", "Enojo", "Miedo", "Sorpresa", "Asco", "Neutral", "Desconocida", "Error: Análisis de emociones fallido", "No especificado"]
        emotion_palette_es = {
            "Alegría": "#4CAF50", "Tristeza": "#2196F3", "Enojo": "#F44336", "Miedo": "#9C27B0",
            "Sorpresa": "#FFC107", "Asco": "#795548", "Neutral": "#9E9E9E", "Desconocida": "#607D8B",
            "Error: Análisis de emociones fallido": "#CCCCCC", "No especificado": "#DDDDDD"
        }
        return plot_comparison_stacked_bar(data_dict, 'emotion', 'Comparación de Emociones por Marca/Tópico',
                                           category_orders_dict={'emotion': emotion_categories_es},
                                           color_discrete_map_dict=emotion_palette_es, x_axis_type='query')

    @output
    @render_plotly
    def same_network_topics_plot():
        data_dict = same_network_comparison_data_r.get() # CAMBIADO
        return plot_topics_comparison_grouped_bar(data_dict, 'Comparación de Tópicos por Marca/Tópico', x_axis_type='query')

    # --- Lógica del Chatbot Comparativo entre Fuentes (Original) ---
    @output
    @render.text # NEW: Separate chat display for cross-source
    def active_session_id_for_cross_source_comparison_chat_display():
        session_id = active_comparison_session_id.get()
        if session_id:
            return f"Contexto del chat: Sesión ID {session_id}"
        return "No hay sesión de comparación activa para el chatbot."
    
    @reactive.Effect
    @reactive.event(input.ask_comparison_chat)
    async def _handle_comparison_chat():
        user_prompt = input.comparison_chat_prompt()
        print(f"DEBUG: ask_comparison_chat triggered. User prompt: '{user_prompt}'") # DEBUG
        session_id = active_comparison_session_id.get()
        current_user = usuario_autenticado.get()

        if not current_user:
            ui.notification_show("Debes iniciar sesión para usar el chatbot.", type="error", duration=5)
            return
        if not session_id:
            ui.notification_show("No hay una sesión de comparación activa para el chatbot.", type="warning", duration=5)
            print("DEBUG: _handle_comparison_chat returning - no session_id") # DEBUG
            return
        if not user_prompt:
            ui.notification_show("Por favor, escribe una pregunta.", type="warning", duration=3)
            print("DEBUG: _handle_comparison_chat returning - no user_prompt") # DEBUG
            return
        if not _ensure_gemini_model():
        #if not _ensure_gemini_model(gemini_instance=gemini_model_instance, gemini_response=current_gemini_response):
            #ui.notification_show("Modelo Gemini no disponible.", type="error", duration=5)
            ui.notification_show("Modelo Gemini no disponible para el chat comparativo.", type="error", duration=5)
            print("DEBUG: _handle_comparison_chat returning - gemini model not ensured") # DEBUG
            return

        gemini_model_chat = gemini_model_instance.get() # Asumimos que es un modelo que soporta chat (historial)
        print("DEBUG: Gemini model instance for chat obtained.") # DEBUG
        
        # Construir contexto de los datos de comparación
        data_for_chat_context = comparison_data_r.get()
        context_text_parts = []
        if data_for_chat_context:
            #for origin, df_source in data_for_chat_context.items():
            for query_label, df_source in data_for_chat_context.items():
                if isinstance(df_source, pd.DataFrame) and not df_source.empty:
                    text_col = None
                    if 'text' in df_source.columns: text_col = 'text'
                    elif 'comment' in df_source.columns: text_col = 'comment'
                    elif 'content' in df_source.columns: text_col = 'content'
                    
                    if text_col:
                        # Tomar una muestra o un resumen breve de cada fuente para el contexto
                        #sample_texts = df_source[text_col].dropna().astype(str).sample(min(len(df_source), 3), random_state=42 if len(df_source) >=3 else None).tolist() 
                        sample_texts = df_source[text_col].dropna().astype(str).sample(min(len(df_source), 10), random_state=42 if len(df_source) >=10 else None).tolist()                        
                        #sample_texts = df_source[text_col].dropna().astype(str).sample(min(len(df_source), 3)).tolist()
                        if sample_texts:
                            #context_text_parts.append(f"Extractos de '{origin}':\n" + "\n".join([f"- {s[:150]}..." for s in sample_texts]))
                            context_text_parts.append(f"Extractos de '{query_label}':\n" + "\n".join([f"- {s[:150]}..." for s in sample_texts]))
        
        full_context_str = "\n\n".join(context_text_parts)
        if not full_context_str:
            full_context_str = "No hay datos específicos cargados para esta sesión de comparación, pero puedes preguntar sobre el resumen general si se generó."
        print(f"DEBUG: Chat context string length: {len(full_context_str)}") # DEBUG

        # Actualizar historial y preparar para la API de Gemini
        #current_history = comparison_chat_history_r.get()
        #current_history.append({"role": "user", "parts": [{"text": user_prompt}]})
        old_history = comparison_chat_history_r.get()
        current_history = old_history + [{"role": "user", "parts": [{"text": user_prompt}]}]

        comparison_chat_history_r.set(current_history) # Actualizar UI con la pregunta del usuario inmediatamente

        # Limitar el tamaño del historial para no exceder límites de la API
        MAX_HISTORY_TURNS = 10 # 5 pares de user/model
        history_for_gemini = current_history.copy()
        if len(history_for_gemini) > MAX_HISTORY_TURNS * 2:
            #history_for_gemini = [history_for_gemini[0]] + history_for_gemini[-(MAX_HISTORY_TURNS * 2 -1):]
            history_for_gemini = history_for_gemini[-(MAX_HISTORY_TURNS * 2):]


        #context_message_for_gemini = {"role": "user", "parts": [f"Considera el siguiente contexto de datos de la sesión de comparación (ID: {session_id[:8]}):\n{full_context_str}\n\nAhora, responde a las preguntas del usuario basándote en este contexto y el historial de conversación."]}
        context_message_for_gemini = {"role": "user", "parts": [f"Considera el siguiente contexto de datos de la sesión de comparación (ID: {session_id[:8] if session_id else 'N/A'}):\n{full_context_str}\n\nAhora, responde a las preguntas del usuario basándote en este contexto y el historial de conversación."]}
        final_content_for_gemini = [context_message_for_gemini] + history_for_gemini

        ui.notification_show("Procesando pregunta con el chatbot comparativo...", type="info", duration=2)
        print("DEBUG: Sending request to Gemini for comparison chat...") # DEBUG
        try:
            # Para modelos como gemini-1.5-flash, pasamos el historial directamente
            # Por ahora, pasaremos el historial completo.
            # response = gemini_model_chat.generate_content(current_history) # Si el modelo soporta historial directamente
            
            # O, si el modelo no maneja bien el historial, construimos un prompt único con el contexto
            # y la pregunta actual, como se hizo para el chat general.
            # Para este ejemplo, vamos a pasar el historial a `generate_content`
            # Asegúrate de que el formato del historial sea el esperado por `generate_content`
            # `contents` (Iterable[glm.Content]): The user's conversational history with the model.
            # Alternativamente, para un solo turno con contexto:
            response = gemini_model_chat.generate_content(final_content_for_gemini)


            #model_response_text = response.text
            #current_history.append({"role": "model", "parts": [{"text": model_response_text}]})
            model_response_text = response.text # Asegúrate que response.text no sea None
            history_after_user_question = comparison_chat_history_r.get() # Obtener el historial que ya incluye la pregunta del user 
            current_history = history_after_user_question + [{"role": "model", "parts": [{"text": model_response_text or "No se recibió respuesta del modelo."}]}]     
            comparison_chat_history_r.set(current_history)
            print("DEBUG: Gemini response received and history updated.") 
        except Exception as e:
            error_msg = f"Error del Chatbot Comparativo: {str(e)}"
            print(error_msg)
            #current_history.append({"role": "model", "parts": [{"text": error_msg}]}) # Añadir error al historial
            #comparison_chat_history_r.set(current_history)
            history_with_error = comparison_chat_history_r.get() + [{"role": "model", "parts": [{"text": error_msg}]}] # Añadir error al historial
            comparison_chat_history_r.set(history_with_error)   
            ui.notification_show(error_msg, type="error", duration=7)

    @output
    @render.ui
    def comparison_chat_response_output():
        history = comparison_chat_history_r.get()
        if not history:
            return ui.p("El historial del chat está vacío. Haz una pregunta.")
        
        chat_html_elements = []
        for msg in history:
            role = msg.get("role")
            #text = msg.get("parts")[0].get("text") if msg.get("parts") and isinstance(msg.get("parts"), list) and msg.get("parts")[0] else "Contenido no disponible"
            # Asegurarse de que 'parts' existe, es una lista y tiene al menos un elemento que es un diccionario con 'text'
            text_part = msg.get("parts")
            text = "Contenido no disponible"
            if isinstance(text_part, list) and len(text_part) > 0 and isinstance(text_part[0], dict) and "text" in text_part[0]:
                text = text_part[0].get("text", "Contenido no disponible")

            if role == "user":
                chat_html_elements.append(ui.div(ui.markdown(f"**Tú:** {text}"), class_="chat-user-message"))
            elif role == "model":
                chat_html_elements.append(ui.div(ui.markdown(text), class_="chat-model-message"))
            chat_html_elements.append(ui.hr(style="margin-top: 5px; margin-bottom: 5px;"))

        return ui.div(*chat_html_elements)

    # --- Lógica del Chatbot Comparativo Misma Red (NEW) ---
    @output
    @render.text # NEW: Separate chat display for same network
    def active_session_id_for_same_network_comparison_chat_display():
        # For same network comparison, there's no session_id from the main app flow.
        # We just indicate it's for the current same-network comparison.
        return "Contexto del chat: Comparación en la misma red"

    @reactive.Effect
    @reactive.event(input.ask_same_network_comparison_chat)
    async def _handle_same_network_comparison_chat():
        user_prompt = input.same_network_comparison_chat_prompt()
        logger.info(f"DEBUG: ask_same_network_comparison_chat triggered. User prompt: '{user_prompt}'")
        current_user = usuario_autenticado.get()

        if not current_user:
            ui.notification_show("Debes iniciar sesión para usar el chatbot.", type="error", duration=5)
            return
        if not user_prompt:
            ui.notification_show("Por favor, escribe una pregunta.", type="warning", duration=3)
            logger.info("DEBUG: _handle_same_network_comparison_chat returning - no user_prompt")
            return
        if not _ensure_gemini_model():
        #if not _ensure_gemini_model(gemini_instance=gemini_model_instance, gemini_response=current_gemini_response):
            ui.notification_show("Modelo Gemini no disponible para el chat comparativo.", type="error", duration=5)
            logger.info("DEBUG: _handle_same_network_comparison_chat returning - gemini model not ensured")
            return

        gemini_model_chat = gemini_model_instance.get()
        logger.info("DEBUG: Gemini model instance for same network chat obtained.")
        
        data_for_chat_context = same_network_comparison_data_r.get()
        context_text_parts = []
        if data_for_chat_context:
            for query_label, df_source in data_for_chat_context.items(): # query_label is the key here
                if isinstance(df_source, pd.DataFrame) and not df_source.empty:
                    text_col = None
                    if 'text' in df_source.columns: text_col = 'text'
                    elif 'comment' in df_source.columns: text_col = 'comment'
                    elif 'content' in df_source.columns: text_col = 'content'
                    
                    if text_col:
                        sample_texts = df_source[text_col].dropna().astype(str).sample(min(len(df_source), 3), random_state=42 if len(df_source) >=3 else None).tolist() 
                        if sample_texts:
                            context_text_parts.append(f"Extractos para '{query_label}':\n" + "\n".join([f"- {s[:150]}..." for s in sample_texts]))
        
        full_context_str = "\n\n".join(context_text_parts)
        if not full_context_str:
            full_context_str = "No hay datos específicos cargados para esta comparación en la misma red."
        logger.info(f"DEBUG: Same network chat context string length: {len(full_context_str)}")

        old_history = same_network_comparison_chat_history_r.get()
        current_history = old_history + [{"role": "user", "parts": [{"text": user_prompt}]}]
        same_network_comparison_chat_history_r.set(current_history)

        MAX_HISTORY_TURNS = 10 
        history_for_gemini = current_history.copy()
        if len(history_for_gemini) > MAX_HISTORY_TURNS * 2:
            history_for_gemini = history_for_gemini[-(MAX_HISTORY_TURNS * 2):]

        context_message_for_gemini = {"role": "user", "parts": [f"Considera el siguiente contexto de datos de la comparación en la misma red:\n{full_context_str}\n\nAhora, responde a las preguntas del usuario basándote en este contexto y el historial de conversación."]}
        final_content_for_gemini = [context_message_for_gemini] + history_for_gemini

        ui.notification_show("Procesando pregunta con el chatbot comparativo...", type="info", duration=2)
        logger.info("DEBUG: Sending request to Gemini for same network comparison chat...")
        try:
            response = gemini_model_chat.generate_content(final_content_for_gemini)
            model_response_text = response.text
            history_after_user_question = same_network_comparison_chat_history_r.get() 
            current_history = history_after_user_question + [{"role": "model", "parts": [{"text": model_response_text or "No se recibió respuesta del modelo."}]}]     
            same_network_comparison_chat_history_r.set(current_history)
            logger.info("DEBUG: Gemini response received and history updated for same network chat.") 
        except Exception as e:
            error_msg = f"Error del Chatbot Comparativo: {str(e)}"
            logger.error(error_msg, exc_info=True)
            history_with_error = same_network_comparison_chat_history_r.get() + [{"role": "model", "parts": [{"text": error_msg}]}] 
            same_network_comparison_chat_history_r.set(history_with_error)   
            ui.notification_show(error_msg, type="error", duration=7)

    @output
    @render.ui # NEW: Separate chat response output for same network
    def same_network_comparison_chat_response_output():
        history = same_network_comparison_chat_history_r.get()
        if not history: return ui.p("El historial del chat está vacío. Haz una pregunta.")
        chat_html_elements = []
        for msg in history:
            role = msg.get("role")
            text_part = msg.get("parts")
            text = "Contenido no disponible"
            if isinstance(text_part, list) and len(text_part) > 0 and isinstance(text_part[0], dict) and "text" in text_part[0]:
                text = text_part[0].get("text", "Contenido no disponible")

            if role == "user": chat_html_elements.append(ui.div(ui.markdown(f"**Tú:** {text}"), class_="chat-user-message"))
            elif role == "model": chat_html_elements.append(ui.div(ui.markdown(text), class_="chat-model-message"))
            chat_html_elements.append(ui.hr(style="margin-top: 5px; margin-bottom: 5px;"))
        return ui.div(*chat_html_elements)
    
    async def fetch_and_analyze_comparison_data_same_network(platform: str, queries: TypingList[str]) -> Dict[str, pd.DataFrame]:
        all_queries_data = {}
        for i, query_text in enumerate(queries):
            if not query_text.strip():
                continue # Skip empty queries
            
            df_query = pd.DataFrame()
            query_label = query_text # Use the query text directly as the label for simplicity in plots

            try:
                if platform == "twitter":
                    df_query = _get_tweets_from_twitter_api(query_text)
                    #df_query = _get_tweets_from_twitter_api_demo(query_text)
                elif platform == "youtube":
                    df_query = _get_youtube_channels_and_comments(query_text, max_results_comments=10) # Max 10 comments per video
                elif platform == "maps":
                    df_query = _maps_comments(query_text)
                # Add other platforms as needed

                if not df_query.empty and not (('Error' in df_query.columns and len(df_query) == 1) or ('Mensaje' in df_query.columns and len(df_query) == 1)):
                    text_col = None
                    if 'text' in df_query.columns: text_col = 'text'
                    elif 'comment' in df_query.columns: text_col = 'comment'
                    elif 'content' in df_query.columns: text_col = 'content'

                    if text_col:
                        df_query['sentiment'] = df_query[text_col].apply(generate_sentiment_analysis)
                        df_query['emotion'] = df_query[text_col].apply(generate_emotions_analysis)
                        collapsed_query_text = collapse_text(df_query)
                        if collapsed_query_text:
                            topics_list_str = topics_generator(collapsed_query_text)
                            try:
                                topics_list = ast.literal_eval(topics_list_str)
                                if isinstance(topics_list, list) and all(isinstance(t, str) for t in topics_list):
                                    df_query['topics'] = topics_list[0] if topics_list else "No especificado"
                                else:
                                    df_query['topics'] = "Error: Formato de tópicos inesperado"
                            except (ValueError, SyntaxError):
                                df_query['topics'] = "Error: Formato de tópicos inválido"
                        else:
                            df_query['topics'] = "No hay texto para tópicos"
                        
                        all_queries_data[query_label] = df_query
                    else:
                        logger.warning(f"No text column found in DataFrame for query '{query_text}' on platform '{platform}'.")
                else:
                    logger.warning(f"No data or error for query '{query_text}' on platform '{platform}'.")
            except Exception as e:
                logger.error(f"Error fetching and analyzing data for query '{query_text}' on platform '{platform}': {e}", exc_info=True)
                all_queries_data[query_label] = pd.DataFrame({'Error': [f"Error al obtener/analizar datos para '{query_text}': {e}"]})
        return all_queries_data

    # --- Lógica para Generador de Infografía (Versión Simplificada en PDF) posteriormente se usará LIDA ---
    @output
    @render.ui
    def infographic_generation_status():
        return ui.p(infographic_status_r.get())

    @render.download(filename=lambda: f"infografia_comparacion_{active_comparison_session_id.get()[:8] if active_comparison_session_id.get() else 'sesion'}.pdf") # ID del handler coincide con el botón
    async def generate_infographic_pdf_handler(): # Nombre del handler cambiado para claridad
        session_id = active_comparison_session_id.get()
        infographic_title = input.infographic_title() if input.infographic_title() else "Infografía de Comparación"

        if not session_id:
            infographic_status_r.set("Error: No hay sesión de comparación activa.")
            yield "Error: No hay sesión de comparación activa."
            return

        data_dict = comparison_data_r.get()
        summary_text = comparison_summary_r.get()

        if not data_dict or not summary_text:
            infographic_status_r.set("Error: No hay datos o resumen de comparación disponibles para generar la infografía.")
            yield "Error: No hay datos o resumen de comparación disponibles."
            return

        infographic_status_r.set("Generando PDF de la infografía...")
        ui.notification_show("Generando PDF...", type="info", duration=10)

        # Crear contenido HTML para el PDF
        html_parts = [f"<html><head><title>{infographic_title}</title><style>body {{ font-family: sans-serif; margin: 20px; }} h1, h2, h3 {{ color: #333; }} .chart-container {{ margin-top: 20px; margin-bottom: 30px; page-break-inside: avoid; }} img {{ max-width: 100%; height: auto; }}</style></head><body>"]
        html_parts.append(f"<h1>{infographic_title}</h1>")
        #html_parts.append(f"<p><i>Sesión ID: {session_id}</i></p>") # type: ignore
        html_parts.append(f"<p><i>Usuario: {usuario_autenticado.get()}</i></p>")
        html_parts.append(f"<h2>Resumen Comparativo General</h2><p>{summary_text.replace('\n', '<br>')}</p>")

        # Gráfico de Sentimientos

        try:
            # Definir categorías y colores si los quieres controlar
            sentiment_categories_pdf = ['Positivo', 'Neutral', 'Negativo',
                                        "Error: Análisis de sentimiento (Pysentimiento) fallido",
                                        "Error: Modelos de sentimiento no disponibles",
                                        "No especificado"]

            sentiment_dfs_list = []
            for origin_name, df_item in data_dict.items():
                if isinstance(df_item, pd.DataFrame) and 'sentiment' in df_item.columns:
                    temp = df_item[['sentiment']].copy()
                    temp['origin'] = origin_name # Asegurar que la columna 'origin' existe
                    sentiment_dfs_list.append(temp)
            
            if sentiment_dfs_list:
                combined_sentiment_df_for_seaborn = pd.concat(sentiment_dfs_list)

                img_bytes_sentiment = generate_seaborn_count_plot_bytes(
                    data_dict=None, # Pasamos el DF combinado directamente
                    data_key_for_df=combined_sentiment_df_for_seaborn, # Pasamos el DF combinado
                    x_column='origin', # Graficar por origen en el eje X
                    hue_column='sentiment', # Diferenciar por sentimiento
                    title='Comparación de Sentimientos por Origen (Seaborn)',
                    category_orders={'sentiment': sentiment_categories_pdf}
                    # color_palette=sentiment_palette_seaborn
                )
                if img_bytes_sentiment:
                    img_base64_sentiment = base64.b64encode(img_bytes_sentiment).decode()
                    html_parts.append("<div class='chart-container'><h2>Sentimientos por Origen</h2>")
                    html_parts.append(f"<img src='data:image/png;base64,{img_base64_sentiment}' alt='Gráfico de Sentimientos (Seaborn)'></div>")
                else:
                    html_parts.append("<p><i>No se pudo generar el gráfico de sentimientos con Seaborn.</i></p>")
            else:
                 html_parts.append("<p><i>No hay datos de sentimiento para graficar con Seaborn.</i></p>")

        except Exception as e_fig_sent_seaborn:
            print(f"Error al generar gráfico de sentimiento con Seaborn: {e_fig_sent_seaborn}")
            html_parts.append("<p><i>Error al generar gráfico de sentimientos (Seaborn) para el PDF.</i></p>")

        html_parts.append("</body></html>")
        html_content_for_pdf = "".join(html_parts)

        html_parts.append("<h2>Puntos Clave (Texto - Gráficos no incluidos en esta versión beta del PDF)</h2>")
        for origin, df_source in data_dict.items():
            if isinstance(df_source, pd.DataFrame) and not df_source.empty:
                html_parts.append(f"<h3>Fuente: {origin}</h3>")
                if 'sentiment' in df_source.columns:
                    sentiment_counts = df_source['sentiment'].value_counts()
                    html_parts.append(f"<p><b>Sentimientos:</b> {sentiment_counts.to_dict()}</p>")
                if 'emotion' in df_source.columns:
                    emotion_counts = df_source['emotion'].value_counts()
                    html_parts.append(f"<p><b>Emociones:</b> {emotion_counts.to_dict()}</p>")
                if 'topics' in df_source.columns:
                    topic_counts = df_source['topics'].value_counts().nlargest(5)
                    html_parts.append(f"<p><b>Tópicos Principales:</b> {topic_counts.to_dict()}</p>")

        html_parts.append("</body></html>")
        html_content_for_pdf = "".join(html_parts)

        try:
            # from weasyprint import HTML
            # pdf_bytes = HTML(string=html_content_for_pdf).write_pdf()
            # infographic_status_r.set("Infografía PDF generada con WeasyPrint. La descarga debería comenzar.")
            # yield pdf_bytes
            from xhtml2pdf import pisa
            pdf_buffer = io.BytesIO()
            # Asegúrate de que html_content_for_pdf sea una cadena codificada en UTF-8 para pisa
            html_content_for_pdf_encoded = html_content_for_pdf.encode("UTF-8")
            src_file_like = io.BytesIO(html_content_for_pdf_encoded)

            pisa_status = pisa.CreatePDF(
                src_file_like,  # un objeto BytesIO del contenido HTML
                dest=pdf_buffer # un objeto BytesIO para almacenar el PDF
            )
            src_file_like.close()

            if pisa_status.err:
                error_msg_pdf = f"Error al generar el PDF con xhtml2pdf: {pisa_status.err}"
                print(error_msg_pdf)
                infographic_status_r.set(error_msg_pdf)
                yield error_msg_pdf.encode('utf-8') # El download handler espera bytes
                return

            pdf_bytes = pdf_buffer.getvalue()
            pdf_buffer.close()
            infographic_status_r.set("Infografía PDF generada con xhtml2pdf. La descarga debería comenzar.")
            yield pdf_bytes
        except ImportError: # Específicamente para xhtml2pdf no encontrado
            error_msg_pdf = "La librería 'xhtml2pdf' no está instalada. Por favor, instálala con 'pip install xhtml2pdf'."
            print(error_msg_pdf)
            infographic_status_r.set(error_msg_pdf)
            yield error_msg_pdf.encode('utf-8')
        except Exception as e_pdf: # Captura otras excepciones durante la generación del PDF
            error_msg_pdf = f"Error al generar el PDF de la infografía con xhtml2pdf: {e_pdf}."
            print(error_msg_pdf)
            infographic_status_r.set(error_msg_pdf)
            yield error_msg_pdf.encode('utf-8')

    # --- Fin Lógica del Módulo de Comparación ---

    @render.download(filename='tabla_scraper_parser.csv')
    async def download_scraper_parser_table():
        results = scraper_parser_results.get()
        df_to_download = None

        if results:
            merged_table_df = results.get("merged_table")
            extracted_tables_list = results.get("extracted_tables", [])

            # Prioridad 1: Tabla fusionada
            if isinstance(merged_table_df, pd.DataFrame) and not merged_table_df.empty and not ('Error' in merged_table_df.columns or 'Mensaje' in merged_table_df.columns):
                df_to_download = merged_table_df
            # Prioridad 2: Primera tabla extraída válida
            elif extracted_tables_list:
                for table_df in extracted_tables_list:
                    if isinstance(table_df, pd.DataFrame) and not table_df.empty and not ('Error' in table_df.columns or 'Mensaje' in table_df.columns):
                        df_to_download = table_df
                        break 
        
        if df_to_download is not None:
            try:
                yield df_to_download.to_csv(index=False, encoding='utf-8-sig')
            except Exception as e:
                print(f"Error al generar CSV para descarga: {e}")
                yield f"Error al generar CSV: {e}"
        else:
            yield "No hay tabla válida para descargar."

    @render.download(filename='tabla_scraper_parser.xlsx')
    async def download_scraper_parser_table_excel():
        results = scraper_parser_results.get()
        df_to_download = None

        if results:
            merged_table_df = results.get("merged_table")
            extracted_tables_list = results.get("extracted_tables", [])

            if isinstance(merged_table_df, pd.DataFrame) and not merged_table_df.empty and not ('Error' in merged_table_df.columns or 'Mensaje' in merged_table_df.columns):
                df_to_download = merged_table_df
            elif extracted_tables_list:
                for table_df in extracted_tables_list:
                    if isinstance(table_df, pd.DataFrame) and not table_df.empty and not ('Error' in table_df.columns or 'Mensaje' in table_df.columns):
                        df_to_download = table_df
                        break 
        
        if df_to_download is not None:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_to_download.to_excel(writer, index=False, sheet_name='Datos')
            yield output.getvalue()
        else:
            yield "No hay tabla válida para descargar en formato Excel."


app = App(app_ui, server)

if __name__ == "__main__":
    app.run()