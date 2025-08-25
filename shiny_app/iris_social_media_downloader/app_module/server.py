# -*- coding: utf-8 -*-
import logging
import pandas as pd
from shiny import render, ui, reactive
from functools import partial
import ast
from pathlib import Path
# --- Módulos del Proyecto ---
from utils import collapse_text, translate_text, detect_language
import psycopg2
from data_fetchers import (
    get_tweets, get_youtube_comments, get_maps_reviews, get_reddit_comments,
    get_playstore_reviews, get_wikipedia_text, get_generic_webpage_text
)
from db_operations import save_df_to_db
from sentiments_and_emotions_classifier.emotions_and_sentiments import generate_emotions_analysis, generate_sentiment_analysis
from llm_models_setup import ModelManager
from plots import create_sentiment_plot, create_emotion_plot, create_topics_plot, generate_interactive_mind_map
from .ui import (ui_login_form, 
                 nav_panel_comparacion_misma_red,
                 nav_panel_analisis_y_visualizaciones, 
                 nav_panel_base_datos_y_chatbot, 
                 nav_panel_comparison_module, 
                 nav_panel_cross_source_comparison,
                 nav_panel_scraper_tablas_chatbot
                 )

from config import PG_HOST, PG_DBNAME, PG_PORT, PG_PASSWORD, PG_USER
logger = logging.getLogger(__name__)

def server(input, output, session):
    model_manager = ModelManager()
    usuario_autenticado = reactive.Value(None)
    processed_dataframe = reactive.Value(pd.DataFrame())
    map_coordinates = reactive.Value(None)
    active_comparison_session_id = reactive.Value(None)
    mensaje_login = reactive.Value("")
    psycopg2_conn_instance = reactive.Value(None)
    @reactive.Calc
    def ui_principal_app():
        return ui.layout_sidebar(
            ui.sidebar(
                ui.output_ui('sidebar_dinamico'),
                width=350
            ),
            ui.navset_card_tab(
                nav_panel_comparacion_misma_red(),
                nav_panel_base_datos_y_chatbot(),
                nav_panel_analisis_y_visualizaciones(),
                nav_panel_comparison_module(),
                nav_panel_cross_source_comparison(),
                nav_panel_scraper_tablas_chatbot(),
                id='pestana_principal_seleccionada'
            )
        )
    

    


    @output 
    @render.ui
    def ui_app_dinamica():
        if usuario_autenticado.get() is None:
            return ui_login_form()
        else:
            return ui_principal_app()

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


    # --- Logos e íconos de la app
    @render.image
    def logo1():
        from shiny.types import ImgData
        dir = Path(__file__).resolve().parent.parent
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
        dir = Path(__file__).resolve().parent.parent
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
        dir = Path(__file__).resolve().parent.parent
        img: ImgData = {
            "src": str(dir / "www/Icon_chismoso_Gemini2.png"), 
            "alt": "Icon",
            "style": "height: 300px; width: auto; display: block; margin-left: auto; margin-right: auto;" # Un poco más pequeño, display block para centrarlo solo
        }        
        return img 
    
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


    # --- Lógica de sesiones para display 
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

    # --- Lógica de Autenticación ---
    # (El código de login y logout se mantiene aquí, ya que es específico de la sesión)

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

 
 
 
    # --- Lógica Principal de la App ---
    @reactive.Effect
    @reactive.event(input.execute)
    def handle_execute():
        platform = input.platform_selector()
        query = get_current_query()
        if not query: return

        df = pd.DataFrame()
        coords = None
        with ui.Progress(min=1, max=10) as p:
            p.set(1, f"Extrayendo datos de {platform}...")
            if platform == "twitter": df = get_tweets(query)
            elif platform == "youtube": df = get_youtube_comments(query)
            elif platform == "maps": df, coords = get_maps_reviews(query)
            elif platform == "reddit": df = get_reddit_comments(query)
            elif platform == "playstore": df = get_playstore_reviews(query)
            elif platform == "wikipedia": df = get_wikipedia_text(query)
            elif platform == "generic_webpage": df = get_generic_webpage_text(query)
            if coords: map_coordinates.set(coords)

            if not df.empty and 'Error' not in df.columns:
                p.set(4, "Traduciendo y analizando...")
                df = process_text_columns(df)
                
                p.set(7, "Guardando en base de datos...")
                if usuario_autenticado.get():
                    save_df_to_db(df, platform, usuario_autenticado.get(), query)

        processed_dataframe.set(df)

    def get_current_query():
        platform = input.platform_selector()
        inputs = {
            "twitter": input.twitter_query,
            "youtube": input.youtube_url,
            "maps": input.maps_query,
            "reddit": input.reddit_url,
            "playstore": input.playstore_url,
            "wikipedia": input.wikipedia_url,
            "generic_webpage": input.generic_webpage_url
        }
        return inputs.get(platform, lambda: "")()

    def process_text_columns(df: pd.DataFrame) -> pd.DataFrame:
        text_col = 'comment' if 'comment' in df.columns else 'text'
        if text_col not in df.columns: return df

        sample_text = collapse_text(df.head())
        lang = detect_language(sample_text, None) # Asumiendo que la API key está en config
        if lang != 'es':
            df[text_col] = df[text_col].apply(lambda x: translate_text(x, lang, 'es'))

        df['sentiment'] = df[text_col].apply(generate_sentiment_analysis)
        df['emotion'] = df[text_col].apply(generate_emotions_analysis)
        
        topics_str = model_manager.topics_generator(collapse_text(df))
        try:
            labels = ast.literal_eval(topics_str)
        except: labels = ["General"]
        
        classify = partial(model_manager.generate_zero_shot_classification_with_labels, candidate_labels=labels)
        df['topics'] = df[text_col].apply(classify)
        return df

    # --- Renderizadores ---
    @output
    @render.data_frame
    def df_data():
        return render.DataGrid(processed_dataframe.get(), height=350)

    @output
    @render.plot
    def sentiment_output():
        return create_sentiment_plot(processed_dataframe.get())

    @output
    @render.plot
    def emotion_plot_output():
        return create_emotion_plot(processed_dataframe.get())

    @output
    @render.plot
    def topics_plot_output():
        return create_topics_plot(processed_dataframe.get())

    @output
    @render.ui
    def mind_map_output():
        html = generate_interactive_mind_map(processed_dataframe.get(), input.platform_selector())
        return ui.tags.iframe(srcDoc=html, width="100%", height="600px") if html else ui.p("Mapa no disponible.")
