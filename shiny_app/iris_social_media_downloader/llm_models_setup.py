# llm_models_setup.py
import logging
import time
from typing import Any, List as TypingList
import json # Necesario para json.dumps
import pandas as pd # Necesario para pd.DataFrame, pd.isna

# Librerías de modelos
import google.generativeai as genai
import spacy
from pysentimiento import create_analyzer
from pinecone import Pinecone, ServerlessSpec
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI as LangchainChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate 

# Importar constantes globales desde config.py
from config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION, OPENROUTER_API_KEY
# Importar funciones de utilidad generales
from utils import collapse_text, markdown_to_csv, split_dom_content # Necesarias para LLM utilities

logger = logging.getLogger(__name__)

# --- Funciones de ensure de LLMs y Pipelines (Necesitan reactive.Value para funcionar) ---
# Estas funciones esperan recibir los reactive.Value correspondientes del server() de Shiny.

def ensure_gemini_embeddings_model_global(gemini_embeddings_model_reactive: Any) -> bool:
    """Asegura que el modelo de embeddings de Gemini esté cargado."""
    if gemini_embeddings_model_reactive.get() is None:
        if GEMINI_API_KEY:
            try:
                logger.info("Iniciando el Gemini 'Embedding'")
                genai.configure(api_key=GEMINI_API_KEY)
                gemini_embeddings_model_reactive.set("models/embedding-001") 
                logger.info("Modelo Gemini 'Embedding' cargado")
                return True
            except Exception as e:
                logger.error(f"Error al configurar Gemini embeddings: {e}", exc_info=True)
                return False    
        else: return False 
    return True 

def ensure_pinecone_client_and_index_global(pinecone_client_reactive: Any, pinecone_index_instance_reactive: Any) -> bool:
    """Asegura que el cliente de Pinecone y el índice estén listos."""
    if pinecone_index_instance_reactive.get() is None: 
        if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
            logger.error('Error: Pinecone API key o indice no configurado.')
            return False
        try:
            logger.info("Se inicia el cliente de Pinecone")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_client_reactive.set(pc)
            if PINECONE_INDEX_NAME not in pc.list_indexes().names():
                logger.info(f"El indice pinecone {PINECONE_INDEX_NAME} no encontrado, creándolo...")
                pc.create_index(name = PINECONE_INDEX_NAME, dimension = EMBEDDING_DIMENSION, 
                                metric= "cosine", spec = ServerlessSpec(cloud ='aws', region = 'us-east-1'))
                while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                    logger.info("Waiting for Pinecone index to be ready...")
                    time.sleep(5)
            else: logger.info('Se encontró el índice de Pinecone')
            index = pc.Index(PINECONE_INDEX_NAME)
            pinecone_index_instance_reactive.set(index)
            logger.info(f"EL índice de Pinecone {PINECONE_INDEX_NAME} se obtuvo.")
            return True 
        except Exception as e:
            logger.error(f"Error al configurar Pinecone: {e}", exc_info=True)
            return False
    return True 

def ensure_llm_model_global(llm_model_instance_reactive: Any) -> bool:
    """Asegura que el modelo LLM para parser de tablas esté cargado."""
    if llm_model_instance_reactive.get() is None:
        if OPENROUTER_API_KEY:
            try:
                logger.info('Iniciando el modelo LLM para parser de tablas')
                model = LangchainChatOpenAI(openai_api_key=OPENROUTER_API_KEY, model="deepseek/deepseek-chat-v3-0324:free", base_url="https://openrouter.ai/api/v1")
                llm_model_instance_reactive.set(model)
                return True
            except Exception as e:
                logger.error(f"Error al cargar el modelo para LLM de DeepSeek: {e}", exc_info=True) 
                return False 
        else: 
            logger.warning("Error: Clave de OpenRouter no configurada para LLM parser de tablas.")
            return False        
    return llm_model_instance_reactive.get() is not None

def ensure_summarizer_pipeline_global(summarizer_pipeline_instance_reactive: Any) -> bool:
    """Asegura que el pipeline de resumen de Hugging Face esté cargado."""
    if summarizer_pipeline_instance_reactive.get() is None:
        try:
            logger.info('Cargando pipeline de resumen')
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summarizer_pipeline_instance_reactive.set(summarizer)
            logger.info('Pipeline de resumen cargado')
            return True
        except Exception as e : 
            logger.error(f"Error al cargar el pipeline de resumen: {e}", exc_info=True)
            return False 
    return True 

def ensure_topics_pipeline_global(topic_pipeline_instance_reactive: Any) -> bool:
    """Asegura que el pipeline de clasificación de tópicos (zero-shot) esté cargado."""
    if topic_pipeline_instance_reactive.get() is None: 
        try: 
            logger.info('Cargando pipeline de tópicos')
            topicGenerator = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            topic_pipeline_instance_reactive.set(topicGenerator)
            logger.info('Pipeline de tópicos cargado')
            return True             
        except Exception as e:
            logger.error(f"Error al cargar el pipeline de tópicos: {e}", exc_info=True)
            topic_pipeline_instance_reactive.set(None) 
            return False 
    return True 
    
def ensure_spacy_sentiment_model_global(spacy_nlp_sentiment_reactive: Any) -> bool:
    """Asegura que el modelo de Spacy para análisis de sentimiento esté cargado."""
    if spacy_nlp_sentiment_reactive.get() is None:
        try:
            logger.info('Iniciando el modelo de Spacy')
            nlp = spacy.load('es_core_news_md')
            if not nlp.has_pipe('spacytextblob'): nlp.add_pipe('spacytextblob')
            spacy_nlp_sentiment_reactive.set(nlp)
            logger.info('Modelo Spacy cargado')
            return True
        except Exception as e:
            logger.error(f"Error al cargar el modelo de Spacy: {e}", exc_info=True)
            return False 
    return True 
    
def ensure_pysentimiento_analyzer_global(pysentimiento_analyzer_instance_reactive: Any) -> bool:
    """Asegura que el analizador de sentimiento de PySentimiento esté cargado."""
    if pysentimiento_analyzer_instance_reactive.get() is None:
        try:
            logger.info('Iniciando el modelo pysentimiento para sentimientos')
            analyzer = create_analyzer(task="sentiment", lang="es")
            pysentimiento_analyzer_instance_reactive.set(analyzer)
            logger.info('Modelo pysentimiento para sentimientos cargado')
            return True
        except Exception as e:
            logger.error(f"Error al cargar el modelo de pysentimiento para sentimientos: {e}", exc_info=True)
            return False
    return True

def ensure_pysentimient_emotions_analyzer_global(pysentimiento_emotions_analyzer_instance_reactive: Any) -> bool:
    """Asegura que el analizador de emociones de PySentimiento esté cargado."""
    if pysentimiento_emotions_analyzer_instance_reactive.get() is None:
        try:
            logger.info('Iniciando el modelo pysentimiento para emociones')
            analyzer = create_analyzer(task="emotion", lang="es")
            pysentimiento_emotions_analyzer_instance_reactive.set(analyzer)
            logger.info('Modelo pysentimiento para emociones cargado')
            return True
        except Exception as e:
            logger.error(f"Error al cargar el modelo de pysentimiento para emociones: {e}", exc_info=True)
            return False
    return True
    
def ensure_gemini_model_global(gemini_model_instance_reactive: Any, current_gemini_response_reactive: Any) -> bool:
    """Asegura que el modelo principal de Gemini esté cargado."""
    if gemini_model_instance_reactive.get() is None:
        if GEMINI_API_KEY:
            try:
                logger.info('Iniciando el modelo de Gemini')
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                gemini_model_instance_reactive.set(model)
                logger.info('Modelo de Gemini inició correctamente')
                return True
            except Exception as e:
                logger.error(f"Error al cargar el modelo de Gemini: {e}", exc_info=True)
                current_gemini_response_reactive.set(f'Error: al iniciar el modelo {e} ') 
                return False 
        else: 
            current_gemini_response_reactive.set("Error: Clave de Gemini no configurada")
            return False 
    return gemini_model_instance_reactive.get() is not None 

# --- LLM Utility functions que requieren modelos reactivos pero son globales ---
# Se asume que collapse_text, markdown_to_csv, split_dom_content son importados de utils.py
def embed_texts_gemini(texts: TypingList[str], gemini_embeddings_model_instance: Any, ensure_embeddings_model_func: Any) -> TypingList[TypingList[float]]:
    """Genera embeddings para una lista de textos usando el modelo de Gemini."""
    if not ensure_embeddings_model_func(): 
        raise Exception('Los embeddings de Gemini no están disponibles.')
    
    model_name = gemini_embeddings_model_instance.get()
    try:
        # task_type está en la API de genai.embed_content, pero no en el type hinting.
        # Se asume que el user desea pasar este task_type.
        result = genai.embed_content(model=model_name, content=texts, task_type="RETRIEVAL_DOCUMENT") # DEFAULT task_type
        return result['embedding']
    except Exception as e:
        logger.error(f"Error al generar los embeddings de Gemini: {e}", exc_info=True)
        return [[] for _ in texts]

def query_pinecone_for_context(query_text: str, top_k: int, ensure_pinecone_func: Any, pinecone_client_instance: Any, pinecone_index_instance: Any, ensure_embeddings_func: Any, gemini_embeddings_model_instance: Any) -> str:
    """Realiza una consulta a Pinecone para obtener contexto."""
    if not ensure_pinecone_func() or not ensure_embeddings_func():
        logger.error('No se pueden obtener la query de Pinecone: Cliente, índice o modelo de embedding no está listo.')
        return ""
    
    try:
        logger.info(f"Insertando la query a Pinecone: '{query_text[:10]}...'")
        # Asegurarse de que task_type sea para 'query'
        query_embedding_list = embed_texts_gemini([query_text], gemini_embeddings_model_instance, ensure_embeddings_func) # No se pasa task_type directamente, usar default
        if not query_embedding_list or not query_embedding_list[0]:
            logger.error("Falló al insertar la query a Pinecone.")
            return ""
        query_embedding = query_embedding_list[0]

        logger.info(f"Llamando a Pinecone con el top_k={top_k}...")
        query_results = pinecone_index_instance.get().query(vector=query_embedding, top_k=top_k, include_metadata=True)
        
        context_parts = [match['metadata']['text'] for match in query_results['matches'] if 'metadata' in match and 'text' in match['metadata']]
        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        logger.error(f"Error al hacer la query a Pinecone: {e}", exc_info=True)
        return ""

def parse_content_with_llm(cleaned_contents: TypingList[str], parse_description: str, ensure_llm_func: Any, llm_model_instance: Any) -> TypingList[str]:
    """Parsea el contenido usando un LLM para extraer información."""
    if not ensure_llm_func(): return ["Error: Modelo LLM no disponible para parseo."]
    model = llm_model_instance
    
    all_raw_outputs = []
    for i, content in enumerate(cleaned_contents):
        if content.startswith("Error scraping"):
            all_raw_outputs.append(f"Error processing content from URL {i+1}: {content}")
            continue
        
        # split_dom_content se asume que está importado de utils
        dom_chunks = split_dom_content(content)
        
        for j, chunk in enumerate(dom_chunks):
            try:
                parse_template = (
                    "You are tasked with extracting specific information from the following text content: {dom_content}. "
                    "Please follow these instructions carefully:\n\n"
                    "1. **Task:** Extract data from the provided text that matches the description: {parse_description}.\n"
                    "2. **Output Format:** Return the extracted data ONLY as one or more Markdown tables. Each table MUST be correctly formatted.\n"
                    "3. **Markdown Table Format:** Each table must adhere to the following Markdown format:\n"
                    "   - Start with a header row, clearly labeling each column, separated by pipes (|).\n"
                    "   - Follow the header row with an alignment row, using hyphens (-) to indicate column alignment (e.g., --- for left alignment).\n"
                    "   - Subsequent rows should contain the data, with cells aligned according to the alignment row.\n"
                    "   - Use pipes (|) to separate columns in each data row.\n"
                    "4. **No Explanations:** Do not include any introductory or explanatory text before or after the table(s).\n"
                    "5. **Empty Response:** If no information matches the description, return an empty string ('').\n"
                    "6. **Multiple Tables:** If the text contains multiple tables matching the description, return each table separately, following the Markdown format for each.\n"
                    "7. **Accuracy:** The extracted data must be accurate and reflect the information in the provided text.\n"
                )
                parse_prompt_template = ChatPromptTemplate.from_template(parse_template)
                response = model.invoke(parse_prompt_template.format_prompt(dom_content=chunk, parse_description=parse_description))
                all_raw_outputs.append(response.content)
                logger.info(f"Parsed chunk {j+1} from content {i+1}")
            except Exception as e:
                logger.error(f"Error parsing chunk {j+1} from content {i+1}: {e}", exc_info=True)
                all_raw_outputs.append(f"Error parsing chunk {j+1} from content {i+1}: {e}")
    return all_raw_outputs

def merge_extracted_tables_llm(tables: TypingList[pd.DataFrame], parse_description: str, ensure_llm_func: Any, llm_model_instance: Any) -> pd.DataFrame:
    """Fusiona tablas extraídas usando un LLM."""
    if not tables: return pd.DataFrame() # Return empty DataFrame if no tables
    if not ensure_llm_func(): return pd.DataFrame({"Error": ["Modelo LLM no disponible para merge."]})
    model = llm_model_instance

    table_strings = [table.to_markdown(index=False) for table in tables]
    
    merge_template = ("You are tasked with merging the following Markdown tables into a single, comprehensive Markdown table.\n"
        "The tables contain information related to: {parse_description}.\n"
        "Please follow these instructions carefully:\n\n"
        "1. **Task:** Merge the data from the following tables into a single table that matches the description: {parse_description}.\n"
        "2. **Output Format:** Return the merged data ONLY as a single Markdown table. The table MUST be correctly formatted.\n"
        "3. **Markdown Table Format:** Each table must adhere to the following Markdown format:\n"
        "   - Start with a header row, clearly labeling each column, separated by pipes (|).\n"
        "   - Follow the header row with an alignment row, using hyphens (-) to indicate column alignment (e.g., --- for left alignment).\n"
        "   - Subsequent rows should contain the data, with cells aligned according to the alignment row.\n"
        "   - Use pipes (|) to separate columns in each data row.\n"
        "4. **No Explanations:** Do not include any introductory or explanatory text before or after the table.\n"
        "5. **Empty Response:** If no information matches the description, return an empty string ('') if no data can be merged.\n"
        "6. **Duplicate Columns:** If there are duplicate columns, rename them to be unique.\n"
        "7. **Missing Values:** If there are missing values, fill them with 'N/A'.\n\n"
        "Here are the tables:\n\n" + "\n\n".join(table_strings) +
        "\n\nReturn the merged table in Markdown format:"
    )
    merge_prompt = ChatPromptTemplate.from_template(merge_template)

    try:
        response = model.invoke(merge_prompt.format_prompt(parse_description=parse_description))
        merged_markdown = response.content
        merged_dfs = markdown_to_csv(merged_markdown)
        if merged_dfs: return merged_dfs[0]
        else:
            logger.warning("LLM merge output did not produce a valid table.")
            return pd.DataFrame({"Mensaje": [f"El LLM intentó fusionar las tablas, pero el resultado no fue una tabla válida. Output del LLM:\n{merged_markdown}"]})
    except Exception as e:
        logger.error(f"Error during LLM merge: {e}", exc_info=True)
        return pd.DataFrame({"Error": [f"Error al fusionar tablas con LLM: {str(e)}"]})
