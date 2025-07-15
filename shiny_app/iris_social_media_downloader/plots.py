# plots.py
import logging
import io
import base64
from typing import Any, Dict, List as TypingList

import pandas as pd
import numpy as np # Necesario si se usa np.nan
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

def plot_sentiment_distribution_seaborn(df_input: pd.DataFrame):
    plt.style.use('seaborn-v0_8-darkgrid')
    sentiment_categories = ['Positivo', 'Neutral', 'Negativo']
    df_input['sentiment'] = pd.Categorical(df_input['sentiment'], categories=sentiment_categories, ordered=True)
    sentiment_counts = df_input['sentiment'].value_counts().reindex(sentiment_categories, fill_value=0)
    palette = {"Positivo": "#2ECC71", "Neutral": "#F1C40F", "Negativo": "#E74C3C"}
    bar_colors = [palette.get(s, '#cccccc') for s in sentiment_counts.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=bar_colors, ax=ax, width=0.6)
    ax.set_title('Análisis de Sentimiento de los Comentarios', fontsize=16, pad=20)
    ax.set_xlabel('Sentimiento', fontsize=14, labelpad=15)
    ax.set_ylabel('Número de Comentarios', fontsize=14, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    total_sentiments = sentiment_counts.sum()
    for i, count in enumerate(sentiment_counts.values):
        if count > 0: 
            percentage = (count / total_sentiments) * 100
            annotation_text = f"{count} ({percentage:.1f}%)"
            ax.text(i, count + (sentiment_counts.max() * 0.015 if sentiment_counts.max() > 0 else 0.15), 
                    annotation_text, ha='center', va='bottom', fontsize=9, color='black')
    plt.tight_layout()
    return fig

def plot_topics_distribution_seaborn(df_input: pd.DataFrame):
    plt.style.use('seaborn-v0_8-darkgrid')
    topics_counts = df_input['topics'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=topics_counts.index, y=topics_counts.values, ax=ax, width=0.6, palette="viridis")
    ax.set_title('Análisis de Tópicos de los Comentarios', fontsize=16, pad=20)
    ax.set_xlabel('Tópicos', fontsize=14, labelpad=15)
    ax.set_ylabel('Número de Comentarios', fontsize=14, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=45, ha="right")
    total_topics = topics_counts.sum()
    for i, count in enumerate(topics_counts.values):
        if count > 0: 
            percentage = (count / total_topics) * 100 if total_topics > 0 else 0
            annotation_text = f"{count} ({percentage:.1f}%)"
            ax.text(i, count + (topics_counts.max() * 0.015 if topics_counts.max() > 0 else 0.15), 
                    annotation_text, ha='center', va='bottom', fontsize=9, color='black')
    plt.tight_layout()
    return fig

def plot_emotion_distribution_seaborn(df_input: pd.DataFrame):
    plt.style.use('seaborn-v0_8-darkgrid') 
    emotion_categories_es = ["Alegría", "Tristeza", "Enojo", "Miedo", "Sorpresa", "Asco", "Neutral", "Desconocida", "Error en análisis"]
    df_input['emotion'] = pd.Categorical(df_input['emotion'], categories=emotion_categories_es, ordered=True)
    emotion_counts = df_input['emotion'].value_counts().reindex(emotion_categories_es, fill_value=0)
    emotion_palette_es = {
        "Alegría": "#4CAF50", "Tristeza": "#2196F3", "Enojo": "#F44336", "Miedo": "#9C27B0",
        "Sorpresa": "#FFC107", "Asco": "#795548", "Neutral": "#9E9E9E", "Desconocida": "#607D8B",
        "Error en análisis": "#BDBDBD"
    }
    bar_colors = [emotion_palette_es.get(cat, '#cccccc') for cat in emotion_counts.index]
    fig, ax = plt.subplots(figsize=(7, 5)) 
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette=bar_colors, ax=ax, width=0.6)
    ax.set_title('Análisis de Emociones de los Comentarios', fontsize=16, pad=20)
    ax.set_xlabel('Emoción', fontsize=14, labelpad=15)
    ax.set_ylabel('Número de Comentarios', fontsize=14, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=45, ha="right")
    total_emotions = emotion_counts.sum()
    for i, count in enumerate(emotion_counts.values):
        if count > 0: 
            percentage = (count / total_emotions) * 100
            annotation_text = f"{count} ({percentage:.1f}%)"
            ax.text(i, count + (emotion_counts.max() * 0.015 if emotion_counts.max() > 0 else 0.15), 
                    annotation_text, ha='center', va='bottom', fontsize=9, color='black')
    plt.tight_layout()
    return fig

def generate_seaborn_count_plot_bytes(data_dict: dict, data_key_for_df: Any, x_column: str, hue_column: str, title: str, category_orders: dict = None, color_palette: dict = None):
    combined_df = None
    if data_dict is None and isinstance(data_key_for_df, pd.DataFrame):
        combined_df = data_key_for_df.copy()
        if not (x_column in combined_df.columns and hue_column in combined_df.columns):
            logger.warning(f"Seaborn: DataFrame provided to data_key_for_df is missing required columns ('{x_column}' or '{hue_column}') for '{title}'.")
            return None
    elif isinstance(data_dict, dict) and data_dict:
        all_dfs_list = []
        for origin_name, df_source_item in data_dict.items():
            if isinstance(df_source_item, pd.DataFrame) and not df_source_item.empty and \
               x_column in df_source_item.columns and hue_column in df_source_item.columns:
                temp_df = df_source_item[[x_column, hue_column]].copy()
                if x_column == 'origin': 
                    temp_df[x_column] = origin_name
                all_dfs_list.append(temp_df)
        
        if not all_dfs_list:
            logger.warning(f"Seaborn: No valid DataFrames found in data_dict to combine for '{title}'.")
            return None
        combined_df = pd.concat(all_dfs_list, ignore_index=True)
    else:
        logger.warning(f"Seaborn: Invalid arguments or no data provided for '{title}'. data_dict type: {type(data_dict)}, data_key_for_df type: {type(data_key_for_df)}")
        return None

    if combined_df is None or combined_df.empty:
        logger.warning(f"Seaborn: No data available to plot for '{title}' after processing inputs.")
        return None
    
    hue_order_list = None
    if category_orders and hue_column in category_orders:
        hue_order_list = category_orders[hue_column]

    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=combined_df,
        x=x_column,
        hue=hue_column,
        order=category_orders.get(x_column) if category_orders else None,
        hue_order=hue_order_list,
        palette=color_palette
    )
    plt.title(title, fontsize=15)
    plt.ylabel("Cantidad", fontsize=12)
    plt.xlabel(x_column.replace('_plot_col','').capitalize(), fontsize=12)
    plt.xticks(rotation=45, ha="right")
    if hue_order_list or combined_df[hue_column].nunique() > 1 :
        plt.legend(title=hue_column.capitalize(), loc='upper right')
    elif plt.gca().get_legend() is not None:
         plt.gca().get_legend().remove()        
    plt.tight_layout()

    img_bytes_io = io.BytesIO()
    plt.savefig(img_bytes_io, format='png', dpi=100)
    plt.close()
    img_bytes_io.seek(0)
    return img_bytes_io.getvalue()

def plot_comparison_stacked_bar(all_data_dict: Dict[str, pd.DataFrame], column_name: str, title: str, category_orders_dict: Dict = None, color_discrete_map_dict: Dict = None, x_axis_type: str = 'origin'):
    plot_data_list = []
    if not all_data_dict: all_data_dict = {}

    for query_key_name, df_source in all_data_dict.items(): 
        if isinstance(df_source, pd.DataFrame) and not df_source.empty and column_name in df_source.columns:
            counts = df_source[column_name].value_counts(dropna=False).reset_index()
            counts.columns = [column_name, 'count']
            counts['query_label'] = query_key_name # Nueva columna para la etiqueta de la barra (marca/tópico)
            plot_data_list.append(counts)

    if not plot_data_list:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (Sin datos)", xaxis_title="Marca/Tópico", yaxis_title="Porcentaje")
        return fig

    combined_df = pd.concat(plot_data_list).fillna({column_name: "No especificado"})
    
    total_counts_per_query = combined_df.groupby('query_label')['count'].sum().reset_index(name='total_query_count')
    combined_df = pd.merge(combined_df, total_counts_per_query, on='query_label', how='left')
    combined_df['percentage'] = (combined_df['count'] / combined_df['total_query_count'].replace(0, 1)) * 100
    combined_df = combined_df.fillna(0)

    fig = px.bar(combined_df, x='query_label', y='percentage', color=column_name,
                 title=title,
                 labels={'query_label': 'Marca/Tópico', 'percentage': 'Porcentaje (%)', column_name: column_name.capitalize()},
                 category_orders=category_orders_dict if category_orders_dict else None,
                 color_discrete_map=color_discrete_map_dict if color_discrete_map_dict else None,
                 text='percentage')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    fig.update_layout(yaxis_ticksuffix='%', barmode='stack', legend_title_text=column_name.capitalize())
    return fig

def plot_topics_comparison_grouped_bar(all_data_dict: Dict[str, pd.DataFrame], title: str):
    plot_data_list = []
    if not all_data_dict: all_data_dict = {}

    for query_key_name, df_source in all_data_dict.items(): # Iterar por query_key_name
        if isinstance(df_source, pd.DataFrame) and not df_source.empty and 'topics' in df_source.columns:
            counts = df_source['topics'].value_counts(dropna=False).reset_index()
            counts.columns = ['topic', 'count']
            counts['query_label'] = query_key_name # Columna para la etiqueta de la marca/tópico
            plot_data_list.append(counts)

    if not plot_data_list: return go.Figure().update_layout(title=f"{title} (Sin datos)", xaxis_title="Tópico", yaxis_title="Conteo")

    combined_df = pd.concat(plot_data_list).fillna({'topic': "No especificado"})
    
    top_n_topics = 15
    overall_topic_counts = combined_df.groupby('topic')['count'].sum().nlargest(top_n_topics).index
    combined_df_filtered = combined_df[combined_df['topic'].isin(overall_topic_counts)]

    fig = px.bar(combined_df_filtered, x='topic', y='count', color='query_label', # Color por marca/tópico
                 title=title,
                 labels={'topic': 'Tópico', 'count': 'Número de Menciones', 'query_label': 'Marca/Tópico'},
                 barmode='group')
    fig.update_xaxes(categoryorder='total descending', tickangle=-45)
    return fig
