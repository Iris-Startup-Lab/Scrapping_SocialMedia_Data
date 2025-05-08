###### Social Media Downloader Shiny App ######
'''It is possible to change the summary method' If is wikipedia or any webpage, use the current method, if is social media, use gemini and simple models like gemma

'''
from shiny import App, render, ui, reactive
import shinyswatch
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import googlemaps
import tweepy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import spacy
import os
import google.generativeai as genai
from chatlas import ChatGoogle
import re 
from spacytextblob.spacytextblob import SpacyTextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
import time
from dotenv import load_dotenv
from pathlib import Path
### Para gráficos
import plotly.express as px
import plotly.graph_objects as go
from scipy.special import softmax
from shinywidgets import output_widget, render_widget, render_plotly
#from crawl4ai import WebCrawler 
## Notas de bugs 
## Funciona todo menos platform inputs


load_dotenv()
### Cargando las variables del ambiente 
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
MAPS_API_KEY = os.environ.get("MAPS_API_KEY")
TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY")
TWITTER_API_SECRET = os.environ.get("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.environ.get("TWITTER_ACCESS_SECRET")
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

#logo_path = os.path.join(os.path.dirname(__file__), "www", "LogoNuevo.png")
here = Path(__file__).parent

#### Comenzando la UI/Frontend
app_ui = ui.page_fixed(
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="styles.css"), # Your existing stylesheet
        ui.tags.link( # Add this for Font Awesome
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" # Or a specific version bundled with your theme
        )
    ),
    #ui.tags.link(rel="stylesheet", href="styles.css"),
    ui.layout_sidebar(
        ui.sidebar(
            #ui.img(src="LogoNuevo.png", style="height: 40px; width: auto; display: block; margin-left: auto; margin-right: auto; margin-bottom: 10px;"),
            #ui.output_image("app_logo", width='100px', height='50px'),
            #ui.img(src="LogoNuevo.png", height='100px', class_="center-block"),
            #ui.img(src="./www/LogoNuevo.png", height='100px', class_="center-block"),
            #ui.img(src="E:/Users/1167486/Local/scripts/Social_media_comments/shiny_app/iris_social_media_downloader/www/LogoNuevo.png", height='100px', class_="center-block"),            
            ui.markdown("**Social Media Downloader** - Extrae y analiza datos de diferentes plataformas."),
            ui.hr(),
            # Selector de plataforma
            ui.input_select(
                "platform_selector",
                "Seleccionar Plataforma:",
                {
                    "wikipedia": "Wikipedia",
                    "youtube": "YouTube",
                    "maps": "Google Maps",
                    "twitter": "Twitter (X)",
                    "generic_webpage": "Página web Genérica"
                }
            ),
            
            # Inputs dinámicos según plataforma seleccionada
            ui.output_ui("platform_inputs"),
            
            #ui.input_action_button("execute", "Ejecutar", class_="btn-primary"),
            ui.input_action_button("execute", "Ejecutar", icon=ui.tags.i(class_="fas fa-play"), class_="btn-primary"),
            width=350
        ),
        
        ui.navset_card_tab(
            ui.nav_panel(
                "Base de datos",
                ui.output_data_frame('df_data'),
                #ui.download_button("download_data", "Descargar CSV", class_="btn-info btn-sm mb-2")
                ui.download_button("download_data", "Descargar CSV", icon=ui.tags.i(class_="fas fa-download"), class_="btn-info btn-sm mb-2 mt-2"),
                icon=ui.tags.i(class_="fas fa-table-list")                
                #ui.output_ui("dynamic_content")
                #ui.output_ui('platform_inputs')
            ),
            ui.nav_panel(
                "Resumen",
                #ui.output_text("summary_output"),
                ui.output_ui('styled_summary_output'),
                icon=ui.tags.i(class_="fas fa-file-lines")

            ),
            ui.nav_panel(
                "Análisis de Sentimiento",
                #output_widget("sentiment_output"),
                ui.output_plot("sentiment_output"),
                #ui.output_ui('styled_summary_output'),
                icon=ui.tags.i(class_="fas fa-file-lines")
            ),
            #ui.nav_panel(
            #    "Análisis de Emociones",
            #    output_widget("emotion_output"),
            #),
            ui.nav_panel(
                "Chat con Gemini",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_text_area("gemini_prompt", "Tu pregunta:", 
                                         placeholder="Escribe tu pregunta para Gemini aquí..."),
                        ui.input_action_button("ask_gemini", "Preguntar", class_="btn-success"),
                        width=350
                    ),
                    ui.card(
                        ui.card_header("Respuesta de Gemini"),
                        ui.output_text("gemini_response"),
                        height="400px",
                        style="overflow-y: auto;"
                    )
                )
            )
        )
    ),
    theme=shinyswatch.theme.darkly()
)

#### Comenzando el server/Backend
def server(input, output, session):
    processed_dataframe = reactive.Value(pd.DataFrame())
    #current_gemini_response = reactive.Value("Escribe tu pregunta y presiona enviar")
    current_gemini_response = reactive.Value("Carga datos y luego haz una pregunta sobre ellos, o haz una pregunta general.")
    gemini_model_instance= reactive.Value(None)
    spacy_nlp_sentiment = reactive.Value(None)
    summarizer_pipeline_instance = reactive.Value(None)
    emotion_model = reactive.Value(None)
    emotion_tokenizer = reactive.Value(None)
    emotion_config = reactive.Value(None)

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
    
    ### Insertamos las funciones generales que se usarán en toda la app 
    def generate_sentiment_analysis(text):
        text = str(text)
        #nlp_sentiment = None
        #try:
        #    nlp_sentiment = spacy.load("es_core_news_md")
        #    nlp_sentiment.add_pipe('spacytextblob')
        #except Exception as e:
        #    print(f"Error al cargar el modelo de Spacy: {e}")
        #    return ui.output_text("sentiment_error", f"Error al cargar el modelo de Spacy: {e}")
        if not _ensure_spacy_sentiment_model():
            return "Error: Modelo de sentimiento no disponible"
        nlp_model = spacy_nlp_sentiment.get()
        if nlp_model is None: 
            return "Error interno modelo de sentimiento"
        #doc = nlp_sentiment(text)
        doc = nlp_model(text)
        polarity = doc._.blob.polarity  
        sentiment='Neutral'
        if polarity > 0.1:
            sentiment = 'Positivo'
        elif polarity < -0.1:
            sentiment = 'Negativo'
        return sentiment

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
                #total_text = df['text']
                #joined_text = " ".join(total_text)
                total_text = df['text'].astype(str)
                joined_text = " ".join(total_text.dropna())
                return joined_text
            elif 'comment' in df.columns:
                #total_text = df['comment']
                #joined_text = " ".join(total_text)
                total_text = df['comment'].astype(str)
                joined_text = " ".join(total_text.dropna())
                return joined_text
            else:
                #return None
                return ""   
        except Exception as e:
            return f"Error al intentar unir el texto: {e}"          


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
    

    def summary_generator(text, platform):   
        #print('Entrando a la función de resumen')
        #summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        #if not _ensure_summarizer_pipeline():
        #    return "Error: Pipelin de resumen no disponible"
        #summarizer = summarizer_pipeline_instance.get()
        if not text:
            return "No hay texto para resumir"
        text = str(text)
        if platform=="wikipedia":
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
    

    @reactive.Effect
    @reactive.event(input.execute)
    def handle_execute():
        platform = input.platform_selector()
        df = pd.DataFrame()
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
        processed_dataframe.set(df)


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
                "twitter_url", 
                "URL de Twitter:", 
                placeholder="https://twitter.com/usuario/status/ID",
                value= '1914271651962700018'
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
        else: 
            return render.DataGrid(pd.DataFrame({"Mensaje": ["Seleccione una plataforma, ingrese los parámetros y presione 'Ejecutar' para cargar datos."]}), height=350)

    @reactive.calc
    def process_wikipedia_for_df():
        if input.platform_selector() != "wikipedia":
            return pd.DataFrame()
        url = input.wikipedia_url()      
        paragraphs = extract_wikipedia_paragraphs(url)
        #if emotion_model.get() is None and not _load_emotion_model():
        #    return pd.DataFrame({'Error': ["No se pudo cargar el modelo de emociones."]})
        data = {'paragraph_number': range(1, len(paragraphs) + 1),
                'text': paragraphs,
                'length': [len(p) for p in paragraphs],
                #'trigrams': [generate_trigrams(p) for p in paragraphs],
                'sentiment': [generate_sentiment_analysis(p) for p in paragraphs]#,
                #'emotion': [detectEmotion(p)[0] for p in paragraphs]
                }
        df = pd.DataFrame(data)
        return df

    @reactive.calc
    def getTweetsResponses():
        if input.platform_selector() != "twitter":
            return pd.DataFrame()
        url = input.twitter_url()      
        twitter_input = url
        if not twitter_input  or not TWITTER_BEARER_TOKEN:
        #if not twitter_input  or not twitter_bearer_token:    
            return pd.DataFrame({'Error': ["URL de Twitter no válida o clave API no configurada."]})
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        if "x.com/" in twitter_input and "/status/" in twitter_input:
            #match = re.search(r'/status/(\d+)/', twitter_input)
            match = re.match(r'.*/status/(\d+)', twitter_input)
            if match:
                tweet_id = match.group(1)
            else:
                return pd.DataFrame({'Error': ["No se pudo extraer el ID del tweet."]})
        elif twitter_input.isdigit():
            tweet_id = twitter_input
        else: 
            return pd.DataFrame({'Error': ["URL de Twitter no válida."]}) 
     
        try: 
            listTweets = client.search_recent_tweets(
                query=f"conversation_id:{tweet_id}",
                expansions=["author_id"],  
                user_fields=["username"],  
                max_results=10
            )

            tweets = listTweets.data
            users = {u["id"]: u for u in listTweets.includes['users']}
            tweets_list = []
            for tweet in tweets:
                tweet_info = {
                    'tweet_id': tweet.id,
                    'text': tweet.text,
                    'author_id': tweet.author_id,
                    'username': users[tweet.author_id]["username"] if tweet.author_id in users else None,
                    'created_at': tweet.created_at if hasattr(tweet, 'created_at') else None
                }
                tweets_list.append(tweet_info)
            df = pd.DataFrame(tweets_list)
            df['sentiment'] = df['text'].apply(generate_sentiment_analysis)
            #df['emotion'] = df['text'].apply(lambda x: detectEmotion(x)[0])
            return pd.DataFrame(df)
        except Exception as e:
            return pd.DataFrame({"Error": [f"Error al obtener comentarios: {str(e)}"]})



    
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
            df['sentiment'] = df['comment'].apply(generate_sentiment_analysis)
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
            return 'Falta el nombre del lugar o no hay una clave válida'
        try:
            comments = []
            gmaps = googlemaps.Client(MAPS_API_KEY)
            find_place_result = gmaps.find_place(placename, input_type='textquery')
            if find_place_result['status']=='OK':
                place_id = find_place_result['candidates'][0]['place_id']
                place_details = gmaps.place(place_id, fields = ['name', 'rating', 'review', 'formatted_address'], language='es')
                reviews = place_details['result'].get('reviews', [])
                while 'next_page_token' in place_details['result']:
                    time.sleep(2)
                    place_details = gmaps.place(place_id, fields=['name', 'rating', 'review', 'formatted_address'], 
                                                page_token=place_details['result']['next_page_token'],  language='es')
                    reviews.extend(place_details['result'].get('reviews', []))
                for review in reviews:
                    comments.append({'author': review['author_name'], 'comment': review['text'], 'rating': review['rating']})
            df = pd.DataFrame(comments)
            df['sentiment'] = df['comment'].apply(generate_sentiment_analysis)
            #df['emotion'] = df['comment'].apply(lambda x: detectEmotion(x)[0])
            return df
        except Exception as e: 
            return pd.DataFrame({"Error": [f"Error al obtener comentarios: {str(e)}"]})

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
        """Generates a styled Seaborn bar chart for sentiment distribution."""
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
        
        # Add value annotations on top of bars
        for i, count in enumerate(sentiment_counts.values):
            if count > 0: # Only annotate if count is greater than 0
                ax.text(i, count + (sentiment_counts.max() * 0.01 if sentiment_counts.max() > 0 else 0.1), 
                        str(count), ha='center', va='bottom', fontsize=11, color='white') # Adjusted for dark theme

        plt.tight_layout()
        return fig

    #@reactive.calc
    def plot_emotion_distribution(df):
        grouped_df= df.groupby('emotion', observed=False).size().reset_index(name='counts')
        barplot = px.bar(grouped_df, x='emotion', y='counts', title='Emotion Analysis of the paragraphs', color='emotion')
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

        final_prompt_to_gemini = f"{data_context}Pregunta del usuario: {user_prompt}"

        print(f"Enviando pregunta a Gemini (con contexto si existe):\n{final_prompt_to_gemini[:50]}...") # Log a snippet        

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
    @render.download(filename='datos_exportados.csv')
    async def download_data():
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
