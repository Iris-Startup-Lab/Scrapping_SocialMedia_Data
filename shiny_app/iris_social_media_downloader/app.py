from shiny import App, render, ui
import shinyswatch

import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import googlemaps
import tweepy
from transformers import pipeline
import spacy
import os
import google.generativeai as genai
import re 
from spacytextblob.spacytextblob import SpacyTextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
MAPS_API_KEY = os.environ.get("MAPS_API_KEY")
TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY")
TWITTER_API_SECRET = os.environ.get("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.environ.get("TWITTER_ACCESS_SECRET")
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


app_ui = ui.page_fixed(
    ui.layout_sidebar(
        ui.sidebar(
            ui.img(src="./LogoNuevo.png", height='200', width='200'),
            ui.markdown("Esta aplicación provisional permite extraer datos de diferentes fuentes y realizar un análisis de sentimientos. Se pueden extraer datos de Wikipedia, YouTube, Google Maps y Twitter (X)."),
            width=200
        )
    ), 
    ui.panel_title("Social Media Downloader", window_title=True),
    ui.navset_tab(
        ui.nav_panel(
            "Wikipedia",
            ui.input_text("wikipedia_url", "URL de Wikipedia:", value="https://es.wikipedia.org/wiki/Python_(lenguaje_de_programaci%C3%B3n)"),
            ui.output_text("wikipedia_content")
        ),
        ui.nav_panel(
            "YouTube",
            ui.input_text("youtube_url", "URL del video de YouTube:", value="https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            ui.output_text("youtube_info"),
            ui.output_text("youtube_comments")
        ),
        ui.nav_panel(
            "Google Maps",
            ui.input_text("maps_query", "Buscar lugar en Google Maps:", value="Ciudad de México"),
            ui.output_text("maps_info"),
            ui.output_text("maps_comments")
        ),
        ui.nav_panel(
            "Twitter (X)",
            ui.input_text("twitter_url", "URL del perfil de Twitter:", value=""),
            ui.output_text("twitter_posts")
        ),
        ui.nav_panel(
            "Resumen",
            ui.output_text("summary_output")
        ),
        ui.nav_panel(
            "Sentimiento",
            #ui.output_text("sentiment_output")
            #ui.output_ui("sentiment_output")
            ui.output_plot("sentiment_output")    
        ),
        ui.nav_panel(
            "Chat con Gemini",
            ui.input_text("gemini_prompt", "Pregunta a Gemini:", value="Cuéntame algo interesante sobre Python."),
            ui.output_text("gemini_response")
        )
    ),
    theme=shinyswatch.theme.darkly

)

def server(input, output, session):
    ''' Extraer contenido de Wikipedia '''
    @output
    @render.text
    def wikipedia_content():
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
    ''' Extraer información de un video de YouTube '''        
    @output
    @render.text
    def youtube_info():
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
    ''' Extraer comentarios de un video de YouTube '''        
    @output
    @render.text
    def youtube_comments():
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
    ''' Extraer información de un sitio con Google Maps '''        
    @output
    @render.text
    def maps_info():
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
    ''' Extraer comentarios de un sitio con Google Maps '''        
    @output
    @render.text
    def maps_comments():
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
    ''' Extraer respuestas a posts de Twitter (X) '''
    @output
    @render.text
    def twitter_posts():
        twitter_input = input.twitter_url()
        if not twitter_input  or not TWITTER_BEARER_TOKEN:
            return "Error: URL de Twitter no válida o clave API no configurada."

        if "twitter.com/" in twitter_input and "/status/" in twitter_input:
            match = re.search(r'/status/(\d+)/', twitter_input)
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

    summarizer = pipeline("summarization", model = "facebook/bart-large-cnn")
    ''' Resumir el contenido de todo'''        
    @output
    @render.text
    def summary_output():
        print("Función summary_output ejecutada") 
        return None
    '''
    def summary_output():
        all_text_to_summarize = ""
        wikipedia_text = input.wikipedia_content()
        print(f"Contenido wikipedia:\n{wikipedia_text[:200]}") 
        youtube_info_text = input.youtube_info()
        print(f"Contenido youtube:\n{youtube_info_text[:200]}")        
        maps_info_text = input.maps_info()
        print(f"Contenido maps:\n{maps_info_text[:200]}")
        twitter_posts_text = input.twitter_posts()
        print(f"Contenido twitter:\n{twitter_posts_text[:200]}")    
        if wikipedia_text and not wikipedia_text.startswith("Error"):
            all_text_to_summarize += wikipedia_text + '\n\n'
        if youtube_info_text and not youtube_info_text.startswith("Error"):
            all_text_to_summarize += youtube_info_text + '\n\n'
        if maps_info_text and not maps_info_text.startswith("Error"):
            all_text_to_summarize += maps_info_text + '\n\n'
        if twitter_posts_text and not twitter_posts_text.startswith("Error"):
            all_text_to_summarize += twitter_posts_text + '\n\n'

        print(f"Texto a resumir:\n{all_text_to_summarize}")
        if all_text_to_summarize:
            try: 
                summary = summarizer(all_text_to_summarize, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
                return f"Resumen:\n{summary}"
            except Exception as e:
                return f"Error al resumir: {e}"
        else:
            return "No hay texto para resumir."
    '''

    ''' Análisis de sentimiento'''  
    @output
    @render.ui
    ##@render.text    

    
    #def sentiment_output():
    #    print("Función sentiment_output ejecutada")
    #    return None
    
    
    def sentiment_output():
        nlp_sentiment = None
        try: 
            nlp_sentiment = spacy.load("es_core_news_md")
            nlp_sentiment.add_pipe('spacytextblob')
        except Exception as e:  
            print(f"Error al cargar el modelo de Spacy: {e}")
            return ui.output_text("sentiment_error", f"Error al cargar el modelo de Spacy: {e}")
            #nlp_sentiment = None

        #if not nlp_sentiment:
            #return "Modelo de NLP no cargado."
            #return ui.output_text("sentiment_error", "Modelo de NLP no cargado.") 
        all_text_to_analyze ={
            "Comentarios de YouTube": input.youtube_comments(),
            "Comentarios de Google Maps": input.maps_comments(),
            "Posts de Twitter": input.twitter_posts(),
            "Texto de Wikipedia": input.wikipedia_content()            
        }
        print(f"Texto a analizar en sentimientos: {all_text_to_analyze}")
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
        fig, ax = plt.subplots(figsize=(10, 6))
        sentiment_counts.plot(kind='bar', stacked=True, ax=ax)
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
        return fig
        #return ui.output_plot(render.plot(fig))
        #return None

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro')
    ''' Chat de Gemini'''
    @output
    @render.text    
    def gemini_response():
        prompt = input.gemini_prompt()
        if not prompt or not GEMINI_API_KEY:
            return "Error: Pregunta no válida o clave API no configurada."
        
        try: 
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al obtener respuesta de Gemini: {e}"
    


app = App(app_ui, server)


if __name__ == "__main__":
    app.run()
