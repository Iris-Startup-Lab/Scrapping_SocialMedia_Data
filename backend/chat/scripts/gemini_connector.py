# backend/chat/scripts/gemini_connector.py
import os
import google.generativeai as genai
from   ..models import RespuestaGemini
from dotenv import load_dotenv

#### Se añade el load dot env para cargar las variables del ambiente
load_dotenv()

### La variable está cargada y es la clave que se puede encontrar en AI Studio de Google Developers
api_gemini = os.getenv('gemini_api_key')

### Se configura el API Key development
genai.configure(api_key=api_gemini)


def obtener_respuesta_gemini(pregunta):
    """ Es una función para las variables de respuesta de Gemini 

     Parámetros
     -----------------
     pregunta : str
         Una cadena con cualquier pregunta para el llm

     Returns/Retorna 
     ----------
     Una cadena simple o compleja de la respuesta del llm
    """
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content(pregunta)
    RespuestaGemini.objects.create(pregunta=pregunta, respuesta=response.text)
    return response.text

if __name__ == "__main__":
        obtener_respuesta_gemini('¿Cuál es el sentimiento general?')
