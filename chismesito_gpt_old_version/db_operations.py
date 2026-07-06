# db_operations.py
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone
import logging

from config import PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD
from utils import get_create_table_sql

logger = logging.getLogger(__name__)

def _get_db_connection():
    """Establece y devuelve una conexión a la base de datos PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, dbname=PG_DBNAME,
            user=PG_USER, password=PG_PASSWORD
        )
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"No se pudo conectar a la base de datos PostgreSQL: {e}")
        return None

def save_df_to_db(df: pd.DataFrame, platform_name: str, requested_by: str, input_reference: str = None, session_id: str = None):
    """Guarda un DataFrame en la tabla de PostgreSQL apropiada."""
    if df.empty or ('Error' in df.columns and len(df) == 1):
        logger.info(f"DataFrame para {platform_name} vacío o con error, no se guarda.")
        return

    table_map = {
        "wikipedia": "wikipedia_data", "youtube": "youtube_comments_data",
        "maps": "maps_reviews_data", "twitter": "twitter_posts_data",
        "generic_webpage": "generic_webpage_data", "reddit": "reddit_comments_data",
        "playstore": "playstore_reviews_data"
    }
    table_name = table_map.get(platform_name)
    if not table_name:
        logger.error(f"No hay tabla definida para la plataforma: {platform_name}")
        return

    conn = _get_db_connection()
    if not conn:
        return

    try:
        with conn.cursor() as cursor:
            df_insert = df.copy()
            if "id" in df_insert.columns: df_insert.drop(columns=["id"], inplace=True)
            
            df_insert["request_timestamp"] = datetime.now(timezone.utc)
            df_insert["requested_by_user"] = requested_by
            if input_reference: df_insert["input_reference"] = input_reference
            if session_id: df_insert["session_id"] = session_id
            
            df_insert.replace({pd.NaT: None, np.nan: None}, inplace=True)

            qualified_table_name = f'iris_scraper."{table_name}"'
            create_stmt = get_create_table_sql(df_insert, qualified_table_name, bool(input_reference), bool(session_id))
            cursor.execute(create_stmt)

            cols = '","'.join(df_insert.columns)
            insert_stmt = f'INSERT INTO {qualified_table_name} ("{cols}") VALUES %s'
            data_tuples = [tuple(x) for x in df_insert.to_numpy()]
            
            execute_values(cursor, insert_stmt, data_tuples)
            conn.commit()
            logger.info(f"{len(data_tuples)} registros guardados en {qualified_table_name}.")
    except Exception as e:
        logger.error(f"Error al guardar en DB para {platform_name}: {e}")
        conn.rollback()
    finally:
        if conn: conn.close()