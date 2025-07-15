##### This script contains functions that were used in some cases but they are not useful anymore 
import pandas as pd 

def _get_tweets_from_twitter_api_demo(query: str) -> pd.DataFrame:
    """
    Generates a demonstration DataFrame for Twitter (X) with a schema that matches
    the real API function (_get_tweets_from_twitter_api).
    Reads data from an Excel file and adapts it.
    """
    print(f"Obteniendo datos de demostración para la consulta: '{query}'")

    try:
        # Assuming 'www' directory is relative to where the script is run
        # If your script is in a different location, adjust 'dir' accordingly
        # For example, if 'www' is in the current working directory:
        #dirdf = Path("www/Demo_datos_a_editar.xlsx")
        # If Path(__file__).resolve().parent works for you, keep it:
        dir = Path(__file__).resolve().parent
        dirdf = str(dir / "www/Demo_datos_a_editar.xlsx")

        datosDemo = pd.read_excel(dirdf, sheet_name='datos_demo_x')
    except Exception as e:
        # If logger is not defined, replace logger.error with print
        # logger.error(f"Error al leer el archivo de demo: {e}")
        print(f"Error al leer el archivo de demo: {e}")
        return pd.DataFrame({'Error': [f"Error al leer el archivo de demo: {e}"]})

    # --- FIX: Remove duplicate columns before processing ---
    # This keeps the first occurrence of any column with a duplicate name
    if datosDemo.columns.duplicated().any():
        print("Advertencia: Se encontraron columnas duplicadas en el archivo Excel. Eliminando duplicados.")
        datosDemo = datosDemo.loc[:, ~datosDemo.columns.duplicated()]
    # --- END FIX ---

    # Filtrar los datos para la consulta específica
    datosDemoFilter = datosDemo[datosDemo['user_query'] == query].copy()

    if datosDemoFilter.empty:
        print(f"No se encontraron datos de demostración para la consulta: '{query}'")
        return pd.DataFrame()

    # Renombrar columnas para que coincidan con el esquema de la API
    # Use a dictionary to map new names to old names
    rename_mapping = {
        'Post Content': 'post_content',
        'Author Name': 'author_name',
        'Post Date': 'created_at',
        'Likes Count': 'like_count',
        'Reposts Count': 'retweet_count',
        'Replies Count': 'reply_count',
        'Views Count': 'impression_count'
    }

    # Apply renaming, only renaming columns that actually exist in the DataFrame
    current_columns = datosDemoFilter.columns.tolist()
    columns_to_rename = {old_name: new_name for old_name, new_name in rename_mapping.items() if old_name in current_columns}
    datosDemoFilter = datosDemoFilter.rename(columns=columns_to_rename)


    # Añadir columnas faltantes con valores simulados
    # Generate random IDs as strings (Twitter IDs are large integers, best handled as strings)
    num_rows = len(datosDemoFilter)
    datosDemoFilter['tweet_id'] = [str(random.randint(10**18, 10**19 - 1)) for _ in range(num_rows)]
    datosDemoFilter['author_id'] = [str(random.randint(10**9, 10**10 - 1)) for _ in range(num_rows)]

    # Ensure 'username' is generated or derived without conflict if 'author_name' exists
    # If 'username' might already be present and derived from 'Author Name' in the Excel,
    # you might need to prioritize or merge. For now, we'll create it directly.
    datosDemoFilter['username'] = datosDemoFilter['author_name'].astype(str).str.replace(" ", "_").str.lower()

    datosDemoFilter['author_verified'] = [random.choice([True, False]) for _ in range(num_rows)]
    datosDemoFilter['quote_count'] = [random.randint(0, 50) for _ in range(num_rows)]
    # Use the newly generated 'tweet_id' for 'conversation_id'
    datosDemoFilter['conversation_id'] = datosDemoFilter['tweet_id']
    datosDemoFilter['in_reply_to_user_id'] = None # No direct reply in this demo for top-level tweets
    datosDemoFilter['lang'] = 'es'
    datosDemoFilter['query'] = query # The original query that found these tweets
    datosDemoFilter['source'] = 'Twitter' # Source platform
    datosDemoFilter['origin'] = 'twitter' # Origin for internal tracking

    # Seleccionar y reordenar columnas para que coincidan exactamente con el esquema final
    final_columns = [
        'tweet_id', 'text', 'author_id', 'username', 'author_name', 'author_verified',
        'created_at', 'like_count', 'retweet_count', 'reply_count', 'quote_count',
        'impression_count', 'conversation_id', 'in_reply_to_user_id', 'lang',
        'query', 'source', 'origin', 'post_content'
    ]

    # Asegurarse de que todas las columnas existan, rellenando con None si es necesario
    # This loop is generally fine, but if a column was duplicated *and then* renamed
    # into a target name, it could cause issues. Cleaning duplicates earlier is key.
    for col in final_columns:
        if col not in datosDemoFilter.columns:
            datosDemoFilter[col] = None

    final_df = datosDemoFilter[final_columns]
    print('Esta es una muestra del dataset original XD')
    print(final_df.head(2))
    final_df = final_df.reset_index(drop=True)
    return final_df
