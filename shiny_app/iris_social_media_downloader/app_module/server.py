# -*- coding: utf-8 -*-
import os
from pathlib import Path
from shiny import App, render, ui, reactive
import shinyswatch
import asyncio
import io
import requests
import time
from dotenv import load_dotenv
import re
import random
import tempfile
import praw
from scipy.special import softmax
import ast
from typing import Any, Dict, List as TypingList
import json
from functools import partial
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget, render_plotly
from plotly.subplots import make_subplots
from PIL import Image
from pyvis.network import Network
import base64
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.chrome.service import Service
from selenium_stealth import stealth
from selenium.webdriver.chrome.options import Options
from google_play_scraper import app as play_app, reviews as play_reviews, Sort, search
import html
import googlemaps
import tweepy
from googleapiclient.discovery import build
import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import google.generativeai as genai
from openai import OpenAI
from langchain_openai import ChatOpenAI as LangchainChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from smolagents import CodeAgent, OpenAIServerModel, Tool
from smolagents.utils import make_image_url, encode_image_base64
from deep_translator import GoogleTranslator, single_detection
import logging
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
import openpyxl
import xlsxwriter
import uuid
from datetime import timedelta
import time

### LLamando a los scripts de ayuda// Calling helper scripts
from ..config import * ### Script con secretos y también ligas a Chrome Driver etc
from ..utils import *
from ..scrapers.webpage_scrapers import _maps_comments, _get_youtube_channels_and_comments, _get_tweets_from_twitter_api
#from ..large_and_small_models.gemini_model_and_functions import _ensure_gemini_model, topics_generator, current_gemini_response, gemini_embeddings_model, gemini_model_instance
from ..sentiments_and_emotions_classifier.emotions_and_sentiments import  spacy_nlp_sentiment, pysentimiento_analyzer_instance, pysentimiento_emotions_analyzer_instance, summarizer_pipeline_instance, emotion_model, emotion_tokenizer, emotion_config
from ..sentiments_and_emotions_classifier.emotions_and_sentiments import _ensure_spacy_sentiment_model,  _ensure_pysentimiento_analyzer, _ensure_pysentimient_emotions_analyzer,  generate_emotions_analysis, sentiment_based_on_emotions_analysis, generate_sentiment_analysis
from ..databases.vector_database_configuration import _ensure_gemini_embeddings_model, _ensure_pinecone_client_and_index, pinecone_index_instance, query_pinecone_for_context
from ..large_and_small_models.gemini_model_and_functions import embed_texts_gemini
from .ui import ui_login_form, nav_panel_base_datos_y_chatbot, nav_panel_analisis_y_visualizaciones, nav_panel_comparison_module, nav_panel_cross_source_comparison, nav_panel_comparacion_misma_red, nav_panel_scraper_tablas_chatbot


################## Mensajes generales y funciones de ayuda ###########
random.seed(42)

print(f"PostgreSQL Host configured: {'Yes' if PG_HOST else 'No'}")
print(f"PostgreSQL User configured: {'Yes' if PG_USER else 'No'}")


# --- End Helper functions ---

def server(input, output, session):
    # Variables reactivas para la autenticación
    usuario_autenticado = reactive.Value(None)
    mensaje_login = reactive.Value("")
    #same_network_comparison_data_r = reactive.Value(None)
    # Configuración inicial de las variables para el server
    ## Lazy Load o Carga del Perezoso
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

    @render.image
    def logo1():
        from shiny.types import ImgData
        dir = Path(__file__).resolve().parent
        img: ImgData = {
            "src": str(dir / "www/Logos_GS_Iris.png"),
            "alt": "Logo 1",
            "style": "height: 70px; width: auto;"
        }        
        return img 
    
    @render.image
    def icon():
        from shiny.types import ImgData
        dir = Path(__file__).resolve().parent
        img: ImgData = {
            "src": str(dir / "www/Icon_chismoso_Gemini2.png"), 
            "alt": "Icon",
            "style": "height: 300px; width: auto; display: block; margin-left: auto; margin-right: auto;"
        }        
        return img 

    @output
    @render.text
    def current_session_id_display():
        session_id = active_comparison_session_id.get()
        if session_id:
            return f"ID Sesión Comparación: {session_id[:8]}..."
        return "No hay sesión de comparación activa."

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
        conn = psycopg2_conn_instance.get()
        cursor = None 
        try:
            cursor = conn.cursor()
            sql_query = "SELECT name, no_employee FROM iris_scraper.iris_email_employees_enabled WHERE email = %s"
            cursor.execute(sql_query, (email.strip().lower(),))
            result = cursor.fetchone()
            if result:
                db_name, db_no_employee = result
                if str(db_no_employee) == password_str:
                    usuario_autenticado.set(db_name if db_name else email.split('@')[0])
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

    @reactive.Effect
    @reactive.event(input.boton_logout)
    def manejar_logout():
        nombre_usuario_actual = usuario_autenticado.get()
        usuario_autenticado.set(None)
        mensaje_login.set("Sesión cerrada exitosamente.")
        if nombre_usuario_actual:
            ui.notification_show(f"Hasta luego, {nombre_usuario_actual}.", type="message", duration=5)
        else:
            ui.notification_show("Sesión cerrada.", type="message", duration=5)

    def fetch_comparison_data(session_id):
        if not _ensure_psycopg2_connection() or not session_id:
            return None
        conn = psycopg2_conn_instance.get()
        all_data = {}
        table_names = [
            "wikipedia_data", "youtube_comments_data", "maps_reviews_data",
            "twitter_posts_data", "generic_webpage_data", "reddit_comments_data",
            "playstore_reviews_data"
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
                        all_data[table] = pd.DataFrame()
            return all_data
        except Exception as e:
            print(f"Error fetching comparison data for session {session_id}: {e}")
            return None

    @render.image
    def app_logo():
        image_path = Path(__file__).parent / "www"/"LogoNuevo.png"
        return {"src": str(image_path), "alt": "App Logo"}

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

    def embed_texts_gemini_(texts: TypingList[str],  task_type="RETRIEVAL_DOCUMENT") -> TypingList[TypingList[float]]:
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
            return result['labels'][0]
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
                total_text = df['comment'].astype(str)
                joined_text = " ".join(total_text.dropna())
                return joined_text
            elif 'content' in df.columns:
                total_text = df['content'].astype(str)
                joined_text = " ".join(total_text.dropna())
                return joined_text
            else:
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
                options.add_argument(f'--proxy-server={proxy}')
            unique_user_data_dir = tempfile.mkdtemp()
            options.add_argument(f"--user-data-dir={unique_user_data_dir}")

            driver_executable_path = None
            prepare_driver_error = None

            if not CHROME_DRIVER_PATH:
                print("CHROME_DRIVER_PATH not set. Using webdriver-manager.")
                try:
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
                abs_driver_path = os.path.abspath(initial_cd_path_env)

                print(f"--- ChromeDriver Path Resolution (CHROME_DRIVER_PATH is set) ---")
                print(f"CHROME_DRIVER_PATH (initial from env): '{initial_cd_path_env}'")
                print(f"Current Working Directory (os.getcwd()): '{os.getcwd()}'")
                print(f"Script Directory (Path(__file__).parent): '{Path(__file__).parent}'")
                print(f"Resolved Absolute Path for chromedriver: '{abs_driver_path}'")
                print(f"--- End ChromeDriver Path Resolution ---")

                driver_executable_path = abs_driver_path

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

            selenium_content = None
            selenium_attempt_error = prepare_driver_error

            if not selenium_attempt_error and driver_executable_path:
                try:
                    print("Initializing Selenium WebDriver...")
                    service = Service(executable_path=driver_executable_path)
                    driver = webdriver.Chrome(service=service, options=options)

                    print(f"Navigating to {website_url} with Selenium...")
                    driver.get(website_url)
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
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(random.uniform(1, 3))
                    print("Page loaded with Selenium.")
                    selenium_content = driver.page_source
                except TimeoutException as e_timeout:
                    selenium_attempt_error = f"Selenium navigation timed out for {website_url}: {str(e_timeout)}"
                    print(selenium_attempt_error)
                except Exception as e_selenium:
                    selenium_attempt_error = f"Selenium WebDriver error (executable: {driver_executable_path}): {str(e_selenium)}"
                    print(selenium_attempt_error)
            elif not selenium_attempt_error and not driver_executable_path:
                 selenium_attempt_error = "ChromeDriver path could not be determined despite no direct error during preparation."
                 print(selenium_attempt_error)

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
                    return selenium_content

            print(f"Attempting fallback with requests for {website_url} (Reason: {selenium_attempt_error or 'Selenium did not yield usable content'})...")
            try:
                agent = {'User-Agent': selected_user_agent}
                if proxy:
                    proxies = {'http': proxy, 'https': proxy}
                    page = requests.get(website_url, headers=agent, timeout=15, proxies=proxies)
                page = requests.get(website_url, headers=agent, timeout=15)
                page.raise_for_status()
                print("Requests fallback successful.")
                return page.content
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
    # ... (rest of the file)
