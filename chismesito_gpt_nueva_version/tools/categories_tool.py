# -*- coding: utf-8 -*-
"""tools/categories_tool.py — LLM genera 5-10 categorías desde muestra 20%."""

import logging
from langchain.tools import tool
from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)


@tool
def generate_categories(comments_text: str) -> dict:
    """
    Genera de 5 a 10 categorias tematicas a partir de una muestra de comentarios.

    El LLM analiza el texto combinado de los comentarios y extrae categorias
    relevantes (maximo 4 palabras cada una).

    Args:
        comments_text: Texto combinado de una muestra (~20%) de los comentarios.

    Returns:
        {"success": bool, "data": [str, str, ...], "count": int, "error": str}
    """
    if not GEMINI_API_KEY:
        return {"success": False, "data": [], "count": 0, "error": "GEMINI_API_KEY no configurada"}

    try:
        from google import genai

        # Truncar para no exceder contexto
        max_len = 7000
        text_to_process = comments_text[:max_len] if len(comments_text) > max_len else comments_text

        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""Analiza los siguientes comentarios extraidos de redes sociales:

{text_to_process}

Genera entre 5 y 10 categorias tematicas (maximo 4 palabras cada una) que
resuman los temas principales. Cada categoria debe ser una etiqueta breve.

IMPORTANTE:
- TODAS las categorias DEBEN estar en ESPAÑOL, sin importar el idioma de los comentarios.
- Responde EXCLUSIVAMENTE con una lista de Python. Ejemplo:
['Calidad del producto', 'Precio', 'Servicio al cliente', 'Entrega', 'Empaque']

No incluyas ningun texto adicional, solo la lista."""
        response = client.models.generate_content(model="gemini-2.0-flash-lite", contents=prompt)

        raw = response.text.strip()
        # Limpiar respuesta
        import ast
        try:
            categories = ast.literal_eval(raw)
            if isinstance(categories, list) and all(isinstance(c, str) for c in categories):
                logger.info(f"Categorias generadas: {categories}")
                return {"success": True, "data": categories, "count": len(categories), "error": None}
            else:
                # Fallback: intentar parsear como texto separado por comas
                cats = [c.strip().strip("'\"") for c in raw.strip("[]").split(",")]
                cats = [c for c in cats if c]
                return {"success": True, "data": cats, "count": len(cats), "error": None}
        except (SyntaxError, ValueError) as e:
            logger.warning(f"No se pudo parsear la respuesta como lista: {raw[:100]}")
            cats = [raw.strip().strip("'\"")]
            return {"success": True, "data": cats, "count": len(cats), "error": None}

    except Exception as e:
        logger.error(f"Categories error: {e}")
        return {"success": False, "data": [], "count": 0, "error": str(e)}
