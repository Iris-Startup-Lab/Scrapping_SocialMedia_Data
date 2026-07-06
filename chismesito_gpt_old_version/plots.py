# plots.py
import logging
import io
import base64
from typing import Any, Dict, List as TypingList

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
import tempfile

logger = logging.getLogger(__name__)

# --- Gráficos para Análisis Individual ---

def create_sentiment_plot(df: pd.DataFrame):
    if df is None or 'sentiment' not in df.columns:
        return None
    plt.figure(figsize=(10, 6))
    sns.countplot(y='sentiment', data=df, palette="viridis")
    plt.title('Distribución de Sentimientos')
    plt.xlabel('Cantidad')
    plt.ylabel('Sentimiento')
    plt.tight_layout()
    return plt.gcf()

def create_emotion_plot(df: pd.DataFrame):
    if df is None or 'emotion' not in df.columns:
        return None
    plt.figure(figsize=(12, 7))
    emotions = df['emotion'].apply(lambda x: x[0] if isinstance(x, list) and x else x)
    sns.countplot(y=emotions, order=emotions.value_counts().index, palette="magma")
    plt.title('Distribución de Emociones')
    plt.xlabel('Cantidad')
    plt.ylabel('Emoción')
    plt.tight_layout()
    return plt.gcf()

def create_topics_plot(df: pd.DataFrame):
    if df is None or 'topics' not in df.columns:
        return None
    plt.figure(figsize=(12, 8))
    sns.countplot(y='topics', data=df, order=df['topics'].value_counts().index, palette="cubehelix")
    plt.title('Distribución de Tópicos')
    plt.xlabel('Cantidad')
    plt.ylabel('Tópicos')
    plt.tight_layout()
    return plt.gcf()

def generate_interactive_mind_map(df: pd.DataFrame, platform_selector: str):
    if df.empty or 'topics' not in df.columns or 'sentiment' not in df.columns:
        return None

    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='in_line')
    central_topic = platform_selector
    net.add_node(central_topic, label=central_topic, color='#007bff', size=25)

    topic_groups = df.groupby('topics')
    for topic, group in topic_groups:
        topic_label = str(topic)
        net.add_node(topic_label, label=topic_label, color='#ffc107', size=15)
        net.add_edge(central_topic, topic_label)

        sentiment_counts = group['sentiment'].value_counts().to_dict()
        for sentiment, count in sentiment_counts.items():
            sentiment_label = f"{sentiment} ({count})"
            sentiment_color = {'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#6c757d'}.get(sentiment, '#adb5bd')
            node_id = f"{topic_label}_{sentiment_label}"
            net.add_node(node_id, label=sentiment_label, color=sentiment_color, size=10)
            net.add_edge(topic_label, node_id)

    try:
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.html', encoding='utf-8') as tmpfile:
            net.save_graph(tmpfile.name)
            tmpfile.seek(0)
            html_content = tmpfile.read()
        return html_content
    except Exception as e:
        logger.error(f"Error generating mind map: {e}")
        return None

# --- Gráficos para Módulos de Comparación ---

def create_comparison_sentiment_plot(data_dict: dict):
    if not data_dict:
        return None
    all_sentiments = []
    for source, df in data_dict.items():
        if 'sentiment' in df.columns:
            sent_counts = df['sentiment'].value_counts(normalize=True).mul(100).rename(source)
            all_sentiments.append(sent_counts)
    if not all_sentiments:
        return None

    sentiment_df = pd.concat(all_sentiments, axis=1).fillna(0).T
    fig = px.bar(sentiment_df, barmode='stack', title="Distribución de Sentimientos por Fuente",
                 labels={'value': 'Porcentaje (%)', 'index': 'Fuente', 'variable': 'Sentimiento'},
                 color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'grey'})
    return fig

def create_comparison_emotion_plot(data_dict: dict):
    if not data_dict:
        return None
    emotion_counts = {}
    for source, df in data_dict.items():
        if 'emotion' in df.columns:
            emotion_counts[source] = df['emotion'].value_counts()
    if not emotion_counts:
        return None

    num_sources = len(emotion_counts)
    fig = make_subplots(rows=1, cols=num_sources, subplot_titles=list(emotion_counts.keys()))
    for i, (source, counts) in enumerate(emotion_counts.items()):
        fig.add_trace(go.Bar(x=counts.index, y=counts.values, name=source), row=1, col=i+1)
    fig.update_layout(title_text="Distribución de Emociones por Fuente", showlegend=False)
    return fig

def create_comparison_topics_plot(data_dict: dict):
    if not data_dict:
        return None
    topic_counts = {}
    for source, df in data_dict.items():
        if 'topics' in df.columns:
            topic_counts[source] = df['topics'].value_counts()
    if not topic_counts:
        return None

    num_sources = len(topic_counts)
    fig = make_subplots(rows=1, cols=num_sources, subplot_titles=list(topic_counts.keys()))
    for i, (source, counts) in enumerate(topic_counts.items()):
        fig.add_trace(go.Bar(x=counts.index, y=counts.values, name=source), row=1, col=i+1)
    fig.update_layout(title_text="Distribución de Tópicos por Fuente", showlegend=False)
    return fig

def create_same_network_sentiment_plot(data_dict: dict):
    if not data_dict:
        return None
    all_sentiments = []
    for query, df in data_dict.items():
        if 'sentiment' in df.columns:
            sent_counts = df['sentiment'].value_counts(normalize=True).mul(100).rename(query)
            all_sentiments.append(sent_counts)
    if not all_sentiments:
        return None

    sentiment_df = pd.concat(all_sentiments, axis=1).fillna(0).T
    fig = px.bar(sentiment_df, barmode='stack', title="Comparación de Sentimientos",
                 labels={'value': 'Porcentaje (%)', 'index': 'Consulta', 'variable': 'Sentimiento'},
                 color_discrete_map={'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#6c757d'})
    return fig

def create_same_network_emotion_plot(data_dict: dict):
    if not data_dict:
        return None
    emotion_counts = {}
    for query, df in data_dict.items():
        if 'emotion' in df.columns:
            emotion_counts[query] = df['emotion'].value_counts()
    if not emotion_counts:
        return None

    num_queries = len(emotion_counts)
    fig = make_subplots(rows=1, cols=num_queries, subplot_titles=list(emotion_counts.keys()))
    for i, (query, counts) in enumerate(emotion_counts.items()):
        fig.add_trace(go.Bar(x=counts.index, y=counts.values, name=query), row=1, col=i+1)
    fig.update_layout(title_text="Comparación de Emociones", showlegend=False)
    return fig