# db_operations.py
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone
import logging
from typing import Any

# Importar constantes globales desde config.py
from config import PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD
# Importar funciones de utilidad desde utils.py
from utils import get_create_table_sql 

logger = logging.getLogger(__name__)

def save_df_to_postgres_with_psycopg2(df: pd.DataFrame, platform_name: str, requested_by: str, input_reference_value: str = None, session_id_value: str = None):
    """Guarda un DataFrame en PostgreSQL usando psycopg2."""
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DBNAME, user=PG_USER, password=PG_PASSWORD)
        logger.info("Conexión a PostgreSQL establecida para guardar datos.")

        table_name_map = {
            "wikipedia": "wikipedia_data", "youtube": "youtube_comments_data", "maps": "maps_reviews_data", 
            "twitter": "twitter_posts_data", "generic_webpage": "generic_webpage_data", "reddit": "reddit_comments_data",
            "playstore": "playstore_reviews_data", "facebook": "facebook_comments_data", # Nueva tabla para Facebook
            "llm_web_parser": "llm_web_parser_data" # Para tablas parseadas con LLM
        }
        table_name = table_name_map.get(platform_name)
        if not table_name:
            logger.error(f"Error: No hay tabla de PostgreSQL definida para '{platform_name}'.")
            return
        if df.empty or ('Error' in df.columns and len(df) == 1) or ('Mensaje' in df.columns and len(df) == 1):
            logger.warning(f"DataFrame para {platform_name} está vacío o es un error/mensaje. No se guarda en PostgreSQL.")
            return
        
        df_insert = df.copy()
        if "id" in df_insert.columns: df_insert = df_insert.drop(columns=["id"])
        df_insert["request_timestamp"] = datetime.now(timezone.utc)
        if input_reference_value is not None: df_insert["input_reference"] = input_reference_value
        if session_id_value is not None: df_insert["session_id"] = session_id_value
        df_insert["requested_by_user"] = requested_by
        df_insert = df_insert.replace({pd.NaT: None, np.nan: None})
        
        qualified_table_name = f"iris_scraper.{table_name}"
        cols_sql = ", ".join([f'"{col_name}"' for col_name in df_insert.columns])
        sql = f"INSERT INTO {qualified_table_name} ({cols_sql}) VALUES %s"
        data_tuples = [tuple(x) for x in df_insert.to_numpy()]

        df_for_schema_definition = df.copy()
        cols_to_drop_for_schema = ["id"]
        if input_reference_value is not None: cols_to_drop_for_schema.append("input_reference")
        if session_id_value is not None: cols_to_drop_for_schema.append("session_id")
        df_for_schema_definition = df_for_schema_definition.drop(columns=cols_to_drop_for_schema, errors='ignore')
        df_for_schema_definition["request_timestamp"] = pd.NaT 
        df_for_schema_definition["requested_by_user"] = ""
        
        cursor = conn.cursor()
        create_table_sql_stmt = get_create_table_sql(
            df_for_schema_definition, qualified_table_name,
            has_input_reference_col=(input_reference_value is not None), 
            has_session_id_col=(session_id_value is not None)
        )            
        logger.info(f"Ensuring table {qualified_table_name} exists...")
        cursor.execute(create_table_sql_stmt)
        execute_values(cursor, sql, data_tuples)
        conn.commit()
        logger.info(f"Se guardaron exitosamente {len(data_tuples)} registros en la tabla '{qualified_table_name}' de PostgreSQL.")
    except psycopg2.Error as e:
        if conn: conn.rollback()
        logger.error(f"Error de Psycopg2 al guardar en '{qualified_table_name}': {e}", exc_info=True)
    except Exception as e: 
        if conn: conn.rollback()
        logger.error(f"Error general al guardar en '{qualified_table_name}': {e}", exc_info=True)        
    finally:
        if cursor: cursor.close()
        if conn: conn.close()
