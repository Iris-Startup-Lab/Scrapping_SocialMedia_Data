# -*- coding: utf-8 -*-
## Iris Startup Lab 
'''
<(*)
  ( >)
  /|
'''
## Fernando Dorantes Nieto

#-------------------------------------------------------------
######### Social Media Downloader Shiny App ######
######### VERSION 0.5 ######
######### Authors Fernando Dorantes Nieto
####### This script hosts some useful functions that help to the app to get several things 
#-------------------------------------------------------------

# utils.py
import pandas as pd
from datetime import datetime, timezone
import re
import ast
from bs4 import BeautifulSoup
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Any, List as TypingList

logger = logging.getLogger(__name__)

def pandas_dtype_to_pg(dtype):
    """Convierte un tipo de dato de Pandas a un tipo de dato PostgreSQL."""
    if pd.api.types.is_integer_dtype(dtype): return "BIGINT"
    elif pd.api.types.is_float_dtype(dtype): return "DOUBLE PRECISION"
    elif pd.api.types.is_bool_dtype(dtype): return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(dtype): return "TIMESTAMP WITHOUT TIME ZONE"
    elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype): return "TEXT"
    else: return "TEXT"

def get_create_table_sql(df_for_schema: pd.DataFrame, qualified_table_name: str, has_input_reference_col: bool, has_session_id_col: bool) -> str:
    """Genera una sentencia SQL CREATE TABLE IF NOT EXISTS a partir de un DataFrame."""
    columns_defs = ["id UUID PRIMARY KEY DEFAULT gen_random_uuid()"]
    if has_input_reference_col: columns_defs.append("input_reference TEXT")
    if has_session_id_col: columns_defs.append("session_id TEXT")
    for col_name, dtype in df_for_schema.dtypes.items():
        pg_type = pandas_dtype_to_pg(dtype)
        columns_defs.append(f'"{col_name}" {pg_type}')
    return f"""CREATE TABLE IF NOT EXISTS {qualified_table_name} (\n    {",\n    ".join(columns_defs)}\n);"""

def collapse_text(df: pd.DataFrame) -> str:
    """Combina el texto de las columnas de texto relevantes de un DataFrame en una sola cadena."""
    try:  # Prioridad: 'text', 'comment', 'content'
        if 'text' in df.columns: total_text = df['text']
        elif 'comment' in df.columns: total_text = df['comment']
        elif 'content' in df.columns: total_text = df['content']
        else: return "" # No hay columna de texto reconocible
        
        joined_text = " ".join(total_text.dropna().astype(str))
        return joined_text
    except Exception as e:
        logger.error(f"Error al intentar unir el texto: {e}", exc_info=True)
        return f"Error al intentar unir el texto: {e}"

def detectLanguage(text: str, api_key: str) -> str:
    """Detecta el idioma de un texto usando la API de detección de idioma."""
    try:
        from deep_translator import single_detection
        language_detected = single_detection(text, api_key=api_key)
        return language_detected
    except Exception as e:
        logger.error(f"Error al detectar el lenguaje: {e}", exc_info=True)
        return 'es' # Fallback a español si falla

def TranslateText(text: str, source: str, target: str) -> str:
    """Traduce texto usando Google Translator."""
    try:
        from deep_translator import GoogleTranslator
        translatedText = GoogleTranslator(source=source, target=target).translate(text)
        return translatedText
    except Exception as e:
        logger.error(f"Error al traducir texto: {e}", exc_info=True)
        return text # Devolver el texto original si falla la traducción

def markdown_to_csv(llm_output: str) -> TypingList[pd.DataFrame]:
    """Convierte tablas formateadas en Markdown a DataFrames de Pandas."""
    tables = re.findall(r"(\|(?:[^\n]+\|)+\n\|(?:\s*-+\s*\|)+\n(?:\|(?:[^\n]+\|)+\n)+)", llm_output)
    dataframes = []
    if tables:
        for table_string in tables:
            try:
                lines = table_string.strip().split("\n")
                if len(lines) >= 3:
                    headers = [col.strip() for col in lines[0].split("|")[1:-1]]  
                    data_rows = [line.split("|")[1:-1] for line in lines[2:]]  
                    cleaned_data = []
                    if headers and not any(h == '' for h in headers):
                        for row in data_rows:
                            if len(row) == len(headers): cleaned_data.append([col.strip() for col in row])
                        else: logger.warning(f"Skipping malformed data row (column count mismatch): {row}")
                        df = pd.DataFrame(cleaned_data, columns=headers)
                        dataframes.append(df)
                    else: logger.warning(f"No valid data rows found for table: {table_string[:100]}...")
            except Exception as e:
                logger.error(f"Error processing a potential Markdown table: {e}\nContent snippet: {table_string[:200]}...", exc_info=True)
                continue
    return dataframes


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
