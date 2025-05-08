###### Social Media Downloader Shiny App ######
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
from scipy.special import softmax


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


app_ui = ui.page_fixed(
    ui.tags.link(rel="stylesheet", href="styles.css"),
    ui.layout_sidebar(
        ui.sidebar(
            ##ui.img(src="LogoNuevo.png", height='100px', class_="center-block"),
            ui.img(src="./www/LogoNuevo.png", height='100px', class_="center-block"),
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
                    "twitter": "Twitter (X)"
                }
            ),
            
            # Inputs dinámicos según plataforma seleccionada
            ui.output_ui("platform_inputs"),
            
            ui.input_action_button("execute", "Ejecutar", class_="btn-primary"),
            width=350
        ),
        
        ui.navset_card_tab(
            ui.nav_panel(
                "Contenido",
                ui.output_data_frame('df_data')
                #ui.output_ui("dynamic_content")
                #ui.output_ui('platform_inputs')
            ),
            ui.nav_panel(
                "Resumen",
                ui.output_text("summary_output")
            ),
            ui.nav_panel(
                "Análisis de Sentimiento",
                ui.output_plot("sentiment_output"),
                ui.output_text("sentiment_text")
            ),
            ui.nav_panel(
                "Análisis de Emociones",
                ui.output_plot("sentiment_output"),
                ui.output_text("sentiment_text")
            ),
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


def server(input, output, session):
    ### Insertamos las funciones generales que se usarán en toda la app 
    @reactive.calc
    def generate_sentiment_analysis(text):
        text = str(text)
        nlp_sentiment = None
        try:
            nlp_sentiment = spacy.load("es_core_news_md")
            nlp_sentiment.add_pipe('spacytextblob')
        except Exception as e:
            print(f"Error al cargar el modelo de Spacy: {e}")
            return ui.output_text("sentiment_error", f"Error al cargar el modelo de Spacy: {e}")
        doc = nlp_sentiment(text)
        polarity = doc._.blob.polarity  
        sentiment='Neutral'
        if polarity > 0.1:
            sentiment = 'Positivo'
        elif polarity < -0.1:
            sentiment = 'Negativo'
        return sentiment

    @reactive.calc
    def detectEmotion(text):
        ### Obtiene el modelo preentrenado
        model_path = "daveni/twitter-xlm-roberta-emotion-es"
        tokenizer = AutoTokenizer.from_pretrained(model_path )
        config = AutoConfig.from_pretrained(model_path )
        emotions_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        ### Starting the encoding
        text = str(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        try:
            output = emotions_model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            emotions_score = np.sort(range(scores.shape[0]))
            emotions_score= emotions_score[0]
            l = config.id2label[ranking[emotions_score]]
            s = scores[ranking[emotions_score]]
            if l=='others':
                l='neutral'
            return l, np.round(float(s), 4)
        except:
            return None, None    
        

    ### Generar los valores de los datos a minar
    collected_data = reactive.Value({
        "wikipedia": "", 
        "youtube": {"info": "", "comments": ""},
        "maps": {"info": "", "comments": ""},
        "twitter": ""
    })


    @reactive.Effect
    @reactive.event(input.execute)
    def execute_scraping():
        platform = input.platform_selector()
        
        with ui.Progress(min=1, max=10) as p:
            p.set(message="Procesando...", detail=f"Extrayendo datos de {platform}")
            print('Comenzando la vista de datos')
            if platform == "wikipedia":
                print('Comenzando la vista de datos, pasó wikipedia')

                data = wikipedia_scrapper()
                current = collected_data.get()
                current["wikipedia"] = data
                collected_data.set(current)
                
            elif platform == "youtube":
                info = youtube_info()
                comments = youtube_comments()
                current = collected_data.get()
                current["youtube"] = {"info": info, "comments": comments}
                collected_data.set(current)
                
            elif platform == "maps":
                info = maps_info()
                comments = maps_comments()
                current = collected_data.get()
                current["maps"] = {"info": info, "comments": comments}
                collected_data.set(current)
                
            elif platform == "twitter":
                posts = twitter_posts()
                current = collected_data.get()
                current["twitter"] = posts
                collected_data.set(current)
    
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
                value = "https://www.youtube.com/watch?v=nwzJpHUGRno"
            )
        elif platform == "maps":
            return ui.input_text(
                "maps_query", 
                "Buscar en Google Maps:", 
                placeholder="Url del sitio o nombre del lugar",
                value = "Av. de los Insurgentes Sur 3579, La Joya, Tlalpan, 14000 Ciudad de México, CDMX"
            )
        elif platform == "twitter":
            return ui.input_text(
                "twitter_url", 
                "URL de Twitter:", 
                placeholder="https://twitter.com/usuario/status/ID"
            )

    @reactive.calc
    def wikipedia_scrapper():
        if input.platform_selector() != "wikipedia":
            return ""
        url = input.wikipedia_url()
        try:
            response = requests.get(url)
            response.raise_for_status()  # Lanza una excepción para códigos de error HTTP
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = "\n".join([p.get_text() for p in paragraphs])
            return text[:500] + "..." if len(text) > 500 else text # Mostrar un fragmento por ahora 
        except requests.exceptions.RequestException as e:
            return f"Error al acceder a Wikipedia: {e}"
    
    @reactive.calc

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


    @output
    @render.data_frame
    def process_wikipedia_for_df():
        if input.platform_selector() != "wikipedia":
            return ""
        url = input.wikipedia_url()      
        paragraphs = extract_wikipedia_paragraphs(url)
        data = {'paragraph_number': range(1, len(paragraphs) + 1),
                'text': paragraphs,
                'length': [len(p) for p in paragraphs],
                'trigrams': [generate_trigrams(p) for p in paragraphs],
                'sentiment': [generate_sentiment_analysis(p) for p in paragraphs],
                'emotion': [detectEmotion(p)[0] for p in paragraphs]
                }
        df = pd.DataFrame(data)
        return df


    @output
    @render.text
    def wikipedia_content():
        return wikipedia_scrapper()

    # ''' Extraer información de un video de YouTube '''        
    @output
    @render.text
    def youtube_info():
        print('Entrando a la función de youtube_info')
        if input.platform_selector() != "youtube":
            return ""
        url = input.youtube_url()
        video_id = None
        if "v=" in url:
            video_id = url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0]
        
        if not video_id or not YOUTUBE_API_KEY:
            return "Error: URL de YouTube no válida o clave API no configurada."
        try:
            youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
            request = youtube.videos().list(part='snippet', id=video_id)
            response = request.execute()
            if response and response['items']:
                item = response['items'][0]["snippet"]
                title = item.get("title", "Sin título")
                description = item.get('description', "Sin descripción")
                return f"Título: {title}\nDescripción: {description[:300]}..."
            else:
                return "No se encontró información del video."
        except Exception as e:
            return f"Error al obtener información del video: {e}"
    # ''' Extraer comentarios de un video de YouTube '''        
    @output
    @render.text
    def youtube_comments():
        print('Entrando a la función de youtube_comments')
        if input.platform_selector() != "youtube":
            return ""
        url = input.youtube_url()
        video_id = None
        if "v=" in url:
            video_id = url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0]
        
        if not video_id or not YOUTUBE_API_KEY:
            return "Error: URL de YouTube no válida o clave API no configurada."
        try:
            youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
            request = youtube.commentThreads().list(part='snippet', videoId=video_id, textFormat='plainText')
            response = request.execute()
            comments = []
            if response and 'items' in response:
                for item in response.get('items', []):
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments.append(comment)
                return "\n".join(comments[:5])  # Mostrar solo los primeros 5 comentarios
            else: 
                return "No se encontraron comentarios."
        except Exception as e:
            return f"Error al obtener comentarios: {e}"
    # ''' Extraer información de un sitio con Google Maps '''        
    @output
    @render.text
    def maps_info():
        if input.platform_selector() != "maps":
            return ""
        query = input.maps_query()
        if not query or not MAPS_API_KEY:
            return "Error: Consulta no válida o clave API no configurada."
        gmaps = googlemaps.Client(key=MAPS_API_KEY)
        try:
            places_results = gmaps.places(query=query)
            if places_results and 'results' in places_results:
                place = places_results['results'][0]
                name = place.get('name', 'Sin nombre')
                address = place.get('formatted_address', 'Sin dirección')
                rating = place.get('rating', 'Sin calificación')
                return f"Nombre: {name}\nDirección: {address}\nCalificación: {rating}"
            else:
                return "No se encontraron resultados."
        except Exception as e:
            return f"Error al obtener información: {e}"
    # ''' Extraer comentarios de un sitio con Google Maps '''        
    @output
    @render.text
    def maps_comments():
        if input.platform_selector() != "maps":
            return ""
        query = input.maps_query()
        if not query or not MAPS_API_KEY:
            return "Error: Consulta no válida o clave API no configurada."            
        gmaps = googlemaps.Client(key=MAPS_API_KEY)
        try: 
            places_result = gmaps.places(query=query)
            if places_result and places_result['results']:
                place_id = places_result['results'][0]['place_id']
                place_details = gmaps.place(place_id=place_id, fields=['reviews'])
                reviews = place_details.get('result', {}).get('reviews', [])
                comments = []
                for review in reviews[:5]:
                    author_name = review.get('author_name', 'Anónimo')
                    text= review.get('text', 'Sin comentario')
                    comments.append(f"{author_name}: {text}")
                return "\n".join(comments) if comments else f"No hay comentarios disponibles para {query}."
            else: 
                return ''
        except Exception as e:
            return f"Error al obtener comentarios: {e}"
    
    # ''' Extraer respuestas a posts de Twitter (X) '''
    @output
    @render.text
    def twitter_posts():
        if input.platform_selector() != "twitter":
            return ""
        twitter_input = input.twitter_url()
        if not twitter_input  or not TWITTER_BEARER_TOKEN:
            return "Error: URL de Twitter no válida o clave API no configurada."

        if "x.com/" in twitter_input and "/status/" in twitter_input:
            #match = re.search(r'/status/(\d+)/', twitter_input)
            match = re.match(r'.*/status/(\d+)', twitter_input)
            if match:
                tweet_id = match.group(1)
            else: 
                return "Error: No se pudo extraer el ID del tweet."
        elif twitter_input.isdigit():
            tweet_id = twitter_input
        else:
            return "Error: URL de Twitter no válida."
        
        if not tweet_id:
            return ''
        try: 
            client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
            listTweets = client.search_recent_tweets(
                query=f"conversation_id:{tweet_id}",
                expansions=["author_id"],
                user_fields=["username"],
                max_results=10
            )
            tweets = listTweets.data
            users = {u["id"]: u for u in listTweets.includes["users"]}
            tweets_list = []
            if tweets: 
                for tweet in tweets:
                    author_username = users.get(tweet.author_id, {}).get('username', 'Desconocido')
                    tweet_info = {
                        'tweet_id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at, 
                        'author_id': tweet.author_id,
                        'author_username': author_username
                    }
                    tweets_list.append(f"- {tweet_info['author_username']}: {tweet_info['text']} (Creado: {tweet_info['created_at']})")
                return "\n".join(tweets_list[:5])    
            else: 
                return "No se encontraron respuestas."
        except Exception as e:
            return f"Error al obtener respuestas: {e}"


    # ''' Resumir el contenido de todo'''        
    @output
    @render.text
    # def summary_output():
    #     print("Función summary_output ejecutada") 
    #     return None    
    def summary_output():
        print('Entrando a la función de resumen')
        summarizer = pipeline("summarization", model = "facebook/bart-large-cnn")
        print('Ha cargado el modelo de resumen')
        #summarizer = pipeline("summarization", model="ELiRF/mt5-base-dacsa-es")
        all_text_to_summarize = ""
        #wikipedia_text = input.wikipedia_content()
        wikipedia_text = wikipedia_scrapper()
        print(f"Contenido wikipedia:\n{wikipedia_text[:200]}") 
        #youtube_info_text = input.youtube_info()
        #print(f"Contenido youtube:\n{youtube_info_text[:200]}")        
        #maps_info_text = input.maps_info()
        #print(f"Contenido maps:\n{maps_info_text[:200]}")
        #twitter_posts_text = input.twitter_posts()
        #print(f"Contenido twitter:\n{twitter_posts_text[:200]}")    
        #print('Uniendo todos los textos para resumir')
        if wikipedia_text and not wikipedia_text.startswith("Error"):
            all_text_to_summarize += wikipedia_text + '\n\n'
        #if youtube_info_text and not youtube_info_text.startswith("Error"):
        #    all_text_to_summarize += youtube_info_text + '\n\n'
        #if maps_info_text and not maps_info_text.startswith("Error"):
        #    all_text_to_summarize += maps_info_text + '\n\n'
        #if twitter_posts_text and not twitter_posts_text.startswith("Error"):
        #    all_text_to_summarize += twitter_posts_text + '\n\n'
        
        print(f"Texto a resumir:\n{all_text_to_summarize}")
        if all_text_to_summarize:
            try: 
                print('Comenzando a resumir')
                summary = summarizer(all_text_to_summarize, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
                return f"Resumen:\n{summary}"
                print('Terminó de resumir')
            except Exception as e:
                return f"Error al resumir: {e}"
        else:
            return "No hay texto para resumir."
    
    @output
    @render.text
    def test_text():
        print("Función test_text ejecutada")
        resultTest = wikipedia_scrapper()
        #return input.wikipedia_content()
        return resultTest
             
    @reactive.calc
    def sentiment_result():
        nlp_sentiment = None
        try:
            nlp_sentiment = spacy.load("es_core_news_md")
            nlp_sentiment.add_pipe('spacytextblob')
        except Exception as e:
            print(f"Error al cargar el modelo de Spacy: {e}")
            return ui.output_text("sentiment_error", f"Error al cargar el modelo de Spacy: {e}")
        all_text_to_analyze = {
            "Test de Wikipedia": wikipedia_scrapper()
        }
        sentiment_data = []
        for source, text in all_text_to_analyze.items():
            if text and not text.startswith("Error"):
                try:
                    doc = nlp_sentiment(text)
                    polarity = doc._.blob.polarity  
                    sentiment='Neutral'
                    if polarity > 0.1:
                        sentiment = 'Positivo'
                    elif polarity < -0.1:
                        sentiment = 'Negativo'
                    sentiment_data.append({"Source": source, "Sentiment": sentiment})
                except Exception as e:
                    print(f"Error analyzing sentiment for {source}: {e}")
            else: 
                print(f"No text available for sentiment analysis for {source}")
        if not sentiment_data:
            return ui.output_text("No hay texto para analizar.")
        
        df_sentiment= pd.DataFrame(sentiment_data)
        sentiment_counts= df_sentiment.groupby('Source')['Sentiment'].value_counts(normalize=True).unstack().fillna(0)
        return sentiment_counts


    @output
    @render.ui
    def dynamic_content():
        platform = input.platform_selector()
        data = collected_data.get()

        if platform == "wikipedia":
            print('Pasó el dynamic content de wikipedia')
            return ui.markdown(data["wikipedia"])
        elif platform == "youtube":
            return ui.markdown(
                f"**Info:**\n{data['youtube']['info']}\n\n**Comments:**\n{data['youtube']['comments']}"
            )
        elif platform == "maps":
            return ui.markdown(
                f"**Info:**\n{data['maps']['info']}\n\n**Comments:**\n{data['maps']['comments']}"
            )
        elif platform == "twitter":
            return ui.markdown(data["twitter"])
        else:
            return ui.markdown("Selecciona una plataforma para ver el contenido.")

    # ''' Análisis de sentimiento'''  
    @output
    @render.plot
    # def sentiment_output():
    #    x= [1,2,3,4,5]
    #    y= [2,3,4,5,6]
    #    fig, ax = plt.subplots()
    #    ax.plot(x,y)
    #    ax.set_title("Ejemplo de gráfico")
    #    ax.set_xlabel("Eje X")
    #    return fig
    def sentiment_output():
        print('Entró en la función de análisis de sentimientos')
        df = sentiment_result()
        print(df.shape)
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind='bar', stacked=True, ax=ax)
        plt.title('Distribución de Sentimientos')
        plt.xlabel("Source")
        plt.ylabel("Percentage")
        plt.xticks(rotation=45, ha="right")

        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height:.1%}', (x + width/2, y + height/2), ha='center', va='center')

        plt.tight_layout()
        print('terminó de graficar')
        return fig
        #return ui.output_plot(render.plot(fig))
        #return None

    # Display functions for each platform
    @output
    @render.text
    def wikipedia_content_display():
        return collected_data.get()["wikipedia"]
    
    @output
    @render.text
    def youtube_info_display():
        return collected_data.get()["youtube"]["info"]
    
    @output
    @render.text
    def youtube_comments_display():
        return collected_data.get()["youtube"]["comments"]
    
    @output
    @render.text
    def maps_info_display():
        return collected_data.get()["maps"]["info"]
    
    @output
    @render.text
    def maps_comments_display():
        return collected_data.get()["maps"]["comments"]
    
    @output
    @render.text
    def twitter_posts_display():
        return collected_data.get()["twitter"]

    # ''' Chat de Gemini'''
    @output
    @render.text    
    def gemini_response():
        print('Entró en la función de respuesta de Gemini')
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')
        prompt = input.gemini_prompt()
        if not prompt or not GEMINI_API_KEY:
            return "Error: Pregunta no válida o clave API no configurada."
        
        try: 
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al obtener respuesta de Gemini: {e}"    
    
    @reactive.Effect
    @reactive.event(input.ask_gemini)
    def ask_gemini_handler():
        print('Entró en la función de respuesta de Gemini')
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')
        prompt = input.gemini_prompt()
        
        if not prompt or not GEMINI_API_KEY:
            return
        
        try:
            with ui.Progress(min=1, max=3) as p:
                p.set(message="Generando respuesta...")
                response = model.generate_content(prompt)
                output.gemini_response.set(response.text)
        except Exception as e:
            print('Entró en la función de respuesta de Gemini')
            output.gemini_response.set(f"Error: {str(e)}")



app = App(app_ui, server)


if __name__ == "__main__":
    app.run()
