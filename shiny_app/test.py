import psycopg2
from datetime import datetime, timedelta
import tweepy
from typing import Optional, Dict, Any
import time


class TwitterAccountManager:
    def __init__(self, db_connection_string: str):
        self.conn = psycopg2.connect(db_connection_string)
    
    def get_available_account(self) -> Optional[Dict[str, Any]]:
        """
        Obtiene una cuenta disponible para usar
        """
        query = """
        SELECT id, api_key, api_secret, access_token, access_token_secret, 
               creation_date, last_used_at_scraper, api_calls, is_available
        FROM twitter_accounts 
        WHERE is_available = TRUE 
        ORDER BY 
            CASE WHEN last_used_at_scraper IS NULL THEN 0 ELSE 1 END,
            last_used_at_scraper ASC,
            api_calls ASC
        LIMIT 1
        """
        
        with self.conn.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
        
        return None
    
    def update_account_after_use(self, account_id: int, success: bool = True):
        """
        Actualiza la cuenta después de usarla
        """
        now = datetime.now()
        
        if success:
            # Añadir 15 minutos para el próximo uso
            next_available = now + timedelta(minutes=15)
            
            update_query = """
            UPDATE twitter_accounts 
            SET 
                last_used_at_scraper = %s,
                api_calls = api_calls + 1,
                is_available = CASE 
                    WHEN (api_calls + 1) >= 100 THEN FALSE 
                    ELSE TRUE 
                END
            WHERE id = %s
            RETURNING api_calls
            """
            
            with self.conn.cursor() as cursor:
                cursor.execute(update_query, (next_available, account_id))
                new_api_calls = cursor.fetchone()[0]
                
                # Si superó el límite de 100 llamadas
                if new_api_calls >= 100:
                    self._handle_rate_limit_exceeded(account_id)
                
                self.conn.commit()
        else:
            # En caso de error, marcar como no disponible
            error_query = """
            UPDATE twitter_accounts 
            SET is_available = FALSE 
            WHERE id = %s
            """
            
            with self.conn.cursor() as cursor:
                cursor.execute(error_query, (account_id,))
                self.conn.commit()
    
    def _handle_rate_limit_exceeded(self, account_id: int):
        """
        Maneja el caso cuando se excede el límite de 100 llamadas
        """
        # Obtener la fecha de creación
        query = "SELECT creation_date FROM twitter_accounts WHERE id = %s"
        
        with self.conn.cursor() as cursor:
            cursor.execute(query, (account_id,))
            creation_date = cursor.fetchone()[0]
            
            # Calcular la fecha de reactivación (1 mes después del mes/día de creación)
            now = datetime.now()
            reactivation_date = datetime(
                now.year if creation_date.month < 12 else now.year + 1,
                (creation_date.month % 12) + 1,
                creation_date.day,
                creation_date.hour,
                creation_date.minute,
                creation_date.second
            )
            
            # Actualizar con la fecha de reactivación
            update_query = """
            UPDATE twitter_accounts 
            SET last_used_at_scraper = %s
            WHERE id = %s
            """
            
            cursor.execute(update_query, (reactivation_date, account_id))
    
    def check_and_reactivate_accounts(self):
        """
        Verifica y reactiva cuentas que ya están disponibles nuevamente
        """
        now = datetime.now()
        
        query = """
        UPDATE twitter_accounts 
        SET is_available = TRUE 
        WHERE 
            (is_available = FALSE AND last_used_at_scraper <= %s)
            OR (api_calls >= 100 AND last_used_at_scraper <= %s)
        """
        
        with self.conn.cursor() as cursor:
            cursor.execute(query, (now, now))
            reactivated_count = cursor.rowcount
            self.conn.commit()
        
        return reactivated_count
    
    def get_twitter_client(self):
        """
        Obtiene un cliente de Tweepy usando una cuenta disponible
        """
        account = self.get_available_account()
        
        if not account:
            # Verificar si hay cuentas que se pueden reactivar
            self.check_and_reactivate_accounts()
            account = self.get_available_account()
            
            if not account:
                raise Exception("No hay cuentas disponibles en este momento")
        
        # Crear cliente de Tweepy
        auth = tweepy.OAuthHandler(account['api_key'], account['api_secret'])
        auth.set_access_token(account['access_token'], account['access_token_secret'])
        
        client = tweepy.Client(
            bearer_token=None,  # Si usas Bearer token, ajústalo
            consumer_key=account['api_key'],
            consumer_secret=account['api_secret'],
            access_token=account['access_token'],
            access_token_secret=account['access_token_secret'],
            wait_on_rate_limit=True
        )
        
        return client, account['id']
    
    def execute_with_account(self, func, *args, **kwargs):
        """
        Ejecuta una función con una cuenta de Twitter disponible
        """
        client, account_id = self.get_twitter_client()
        
        try:
            result = func(client, *args, **kwargs)
            self.update_account_after_use(account_id, success=True)
            return result
        
        except Exception as e:
            self.update_account_after_use(account_id, success=False)
            raise e
    
    def close(self):
        """Cierra la conexión a la base de datos"""
        self.conn.close()

# Ejemplo de uso:
def ejemplo_uso_twitter(client, query: str):
    """
    Función de ejemplo que usa el cliente de Twitter
    """
    # Ejemplo: buscar tweets
    tweets = client.search_recent_tweets(
        query=query,
        max_results=10,
        tweet_fields=['created_at', 'author_id']
    )
    return tweets.data

# Uso del manager
def main():
    # Cadena de conexión a Supabase
    db_connection_string = "postgresql://username:password@host:port/database"
    
    manager = TwitterAccountManager(db_connection_string)
    
    try:
        # Ejecutar una operación con una cuenta disponible
        resultados = manager.execute_with_account(
            ejemplo_uso_twitter, 
            "python programming"
        )
        
        print(f"Resultados obtenidos: {len(resultados)} tweets")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        manager.close()

if __name__ == "__main__":
    main()