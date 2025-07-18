import os 
import sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from shiny import  reactive
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
from shiny import ui, reactive


from config import SUPABASE_URL, SUPABASE_KEY, PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD


options = ClientOptions().replace(schema="iris_scraper") # 
client = create_client(SUPABASE_URL, SUPABASE_KEY, options=options)
supabase_client_instance = reactive.Value(None)
psycopg2_conn_instance = reactive.Value(None)

# ----- Supabase ensure psql
def _ensure_psycopg2_connection():
    if psycopg2_conn_instance.get() is None:
        if not all([PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD]):
            print("Error: PostgreSQL connection details not fully configured.")
            ui.notification_show("PostgreSQL no configurado. No se guardarán los datos.", type="error", duration=7)
            return False 
        try:
            print("Attempting to connect to PostgreSQL...")
            conn = psycopg2.connect(
                host = PG_HOST,
                port = PG_PORT,
                dbname = PG_DBNAME,
                user = PG_USER,
                password = PG_PASSWORD
            )
            psycopg2_conn_instance.set(conn)
            print('Se conectó a PostgreSQL exitosamente')
            return True 
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}") 
            ui.notification_show(f"Error al conectar PostgreSQL: {e}", type="error", duration=7)
            return False 
    return True    



