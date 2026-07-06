# -*- coding: utf-8 -*-
## Iris Startup Lab 
'''
<(*)
  ( >)
  /|
'''
### Fernando Dorantes Nieto
## Fecha de debuggeo: 2025-07-21

#### Pinecone configuration
###### NOTA faltan los embeddings de Gemini !!!!!!
import sys 
import os 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pinecone import Pinecone, ServerlessSpec
from shiny import reactive
import traceback
import time 
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION
from large_and_small_models.gemini_model_and_functions import  _ensure_gemini_embeddings_model, embed_texts_gemini


pinecone_client = reactive.Value(None)
pinecone_index_instance = reactive.Value(None)



def _ensure_pinecone_client_and_index():
    if pinecone_index_instance.get() is None: 
        if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
            print('Error: Pinecone API o indice no encontrados ')
            return False
        try:
            print("Se inicia el cliente de Pinecone")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_client.set(pc)
            if PINECONE_INDEX_NAME not in pc.list_indexes().names():
                print(f"El indice pinecone {PINECONE_INDEX_NAME} no encontrado")
                pc.create_index(
                    name = PINECONE_INDEX_NAME,
                    dimension = EMBEDDING_DIMENSION, 
                    metric= "cosine", 
                    spec = ServerlessSpec(
                        cloud ='aws', ### por default esta es la única opción disponible en el modelo gratuito
                        region = 'us-east-1' 
                    )
                )
                while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                     print('Esperando que el índice de Pinecone esté listo')
                     time.sleep(5)
            else: 
                 print('Se encontró el índice')
                 index = pc.Index(PINECONE_INDEX_NAME)
                 pinecone_index_instance.set(index)
                 return True 
        except Exception as e:
            print(f"Error al configurar Pinecone {e}")
            traceback.print_exc()
            return False
    return True


def query_pinecone_for_context(query_text: str, top_k: int = 3) -> str: 
    #if _ensure_pinecone_client_and_index():
    if not _ensure_pinecone_client_and_index() or not _ensure_gemini_embeddings_model():    
        print("No se pueden obtener la query de Pinecone: Cliente, índice o modelo de embedding no está listo")
        return ""
    try:
        print(f"Insertandoa la query a Pinecone: {query_text[:10]}....")
        query_embedding_list = embed_texts_gemini([query_text], task_type="RETRIEVAL_QUERY")
        if not query_embedding_list or not query_embedding_list[0]:
            print("Falló al insertar la query a Pinecone")
            return ""
        query_embedding = query_embedding_list[0]
        print(f"LLamando a Pinecone con el top_k={top_k}...")
        query_results = pinecone_index_instance.get().query(vector=query_embedding, top_k=top_k, include_metadata=True)
        context_parts = [match['metadata']['text'] for match in query_results['matches'] if 'metadata' in match and 'text' in match['metadata']]
        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        print(f"Error al hacer la query a Pinecone: {e}")
        return ""
