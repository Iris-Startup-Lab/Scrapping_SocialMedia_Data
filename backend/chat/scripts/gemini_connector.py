# backend/chat/scripts/gemini_connector.py
import google.generativeai as genai
from   ..models import RespuestaGemini

genai.configure(api_key="TU_API_KEY_GEMINI")

def obtener_respuesta_gemini(pregunta):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(pregunta)
    RespuestaGemini.objects.create(pregunta=pregunta, respuesta=response.text)
    return response.text

if __name__ == "__main__":
    obtener_respuesta_gemini('¿Cuál es el sentimiento general?')