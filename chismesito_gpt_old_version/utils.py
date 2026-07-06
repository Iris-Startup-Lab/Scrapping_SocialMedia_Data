# utils.py
import pandas as pd
import logging
from deep_translator import GoogleTranslator, single_detection

logger = logging.getLogger(__name__)

def pandas_dtype_to_pg(dtype):
    """Convierte un tipo de dato de Pandas a un tipo de dato PostgreSQL."""
    if pd.api.types.is_integer_dtype(dtype): return "BIGINT"
    elif pd.api.types.is_float_dtype(dtype): return "DOUBLE PRECISION"
    elif pd.api.types.is_bool_dtype(dtype): return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(dtype): return "TIMESTAMP WITHOUT TIME ZONE"
    else: return "TEXT"

def get_create_table_sql(df_for_schema: pd.DataFrame, qualified_table_name: str, has_input_reference_col: bool, has_session_id_col: bool) -> str:
    """Genera una sentencia SQL CREATE TABLE IF NOT EXISTS a partir de un DataFrame."""
    columns_defs = ["id UUID PRIMARY KEY DEFAULT gen_random_uuid()"]
    if has_input_reference_col: columns_defs.append("input_reference TEXT")
    if has_session_id_col: columns_defs.append("session_id TEXT")
    for col_name, dtype in df_for_schema.dtypes.items():
        pg_type = pandas_dtype_to_pg(dtype)
        columns_defs.append(f'"{col_name}" {pg_type}')
    return f"""CREATE TABLE IF NOT EXISTS {qualified_table_name} (\n    {',\n    '.join(columns_defs)}\n);"""

def collapse_text(df: pd.DataFrame) -> str:
    """Combina el texto de las columnas de texto relevantes de un DataFrame."""
    text_col = None
    if 'text' in df.columns: text_col = 'text'
    elif 'comment' in df.columns: text_col = 'comment'
    elif 'content' in df.columns: text_col = 'content'
    
    if text_col:
        return " ".join(df[text_col].dropna().astype(str))
    return ""

def detect_language(text: str, api_key: str) -> str:
    """Detecta el idioma de un texto."""
    try:
        return single_detection(text, api_key=api_key)
    except Exception as e:
        logger.warning(f"Detección de idioma falló: {e}. Usando 'es' como fallback.")
        return 'es'

def translate_text(text: str, source: str, target: str = 'es') -> str:
    """Traduce texto de un idioma a otro."""
    if not text or source == target:
        return text
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception as e:
        logger.error(f"Error de traducción de {source} a {target}: {e}")
        return text # Devuelve el original si falla