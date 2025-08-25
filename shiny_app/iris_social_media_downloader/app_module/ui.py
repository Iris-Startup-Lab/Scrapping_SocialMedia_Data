# -*- coding: utf-8 -*-
from shiny import ui
import shinyswatch
from pathlib import Path
from shinywidgets import output_widget

# Definición principal de la UI, importada por app.py
app_ui = ui.page_fixed(
    ui.tags.head(
        ui.tags.link( 
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
        ),
        ui.tags.script("""
            $(document).ready(function(){
                $('#gemini_prompt').on('keypress', function(e){
                    if(e.which == 13 && !e.shiftKey){
                        e.preventDefault();
                        $('#ask_gemini').click();
                    }
                });
                $('#email_login, #password_login').on('keypress', function(e){
                    if(e.which == 13){
                        e.preventDefault();
                        $('#boton_login').click();
                    }
                });
            });
        """)
    ),
    ui.output_ui("ui_app_dinamica"),
    theme=shinyswatch.theme.darkly()
)

# --- Componentes de UI reutilizables ---

def ui_login_form():
    """Retorna la UI para el formulario de login."""
    return ui.div(
        ui.row(
            ui.column(4,
                      ui.panel_well(
                          ui.hr(),
                          ui.h1("Iris Startup Lab", style="text-align: center; font-size: 2em;"), 
                          ui.h1("Presenta:", style="text-align: center; font-size: 0.8em;"), 
                          ui.h1('"ChismesitoGPT"', style="text-align: center; font-size: 1.5em; font-weight: bold;"), 
                          ui.h1("Obten datos de la web y redes sociales + IA", style="text-align: center; font-size: 0.8em;"), 
                          ui.div(
                              ui.output_image("icon"),
                              style="text-align: center;", class_="login-icon-container" 
                          ),
                          ui.h3("Acceso:", style="text-align: center; font-size: 1.2em;"), 
                          ui.hr(),
                          ui.input_text("email_login", "Correo Electrónico:", placeholder="tucorreoelectronico@elektra/dialogus"),
                          ui.input_password("password_login", "Contraseña (No. Empleado):", placeholder="Tu número de empleado"),                              
                          ui.input_action_button("boton_login", "Ingresar", class_="btn-primary btn-block mt-2"),
                          ui.output_text("texto_mensaje_login"),
                          ui.hr(), 
                          ui.div(
                              ui.div(ui.output_image("logo1"), 
                                     style="display: inline-block; vertical-align: middle; margin-right: 10px;"),
                              style="text-align: center;", class_="login-logo-container" 
                          ),                              
                          style="color: #00968b; margin-top: 10px; text-align: center;"
                      ),
                    offset=4,
                    class_="login-panel-column"
            )
        ),
        style="margin-top: 100px;"
    )

def nav_panel_base_datos_y_chatbot():
    return ui.nav_panel(
        "Base de Datos y Chatbot",
        ui.navset_card_pill(
            ui.nav_panel("Base de Datos", 
                         ui.output_data_frame('df_data'),
                         ui.download_button("download_data", "Descargar CSV", icon=ui.tags.i(class_="fas fa-download"), class_="btn-info btn-sm mb-2 mt-2"),
                         ui.download_button("download_data_excel", "Descargar Excel", icon=ui.tags.i(class_="fas fa-file-excel"), class_="btn-success btn-sm mb-2 mt-2 ms-2"),
                         icon=ui.tags.i(class_="fas fa-table-list")),
            ui.nav_panel("Resumen General", 
                         ui.output_ui('styled_summary_output'), 
                         icon = ui.tags.i(class_ ="fa-solid fa-newspaper")),
            ui.nav_panel("Chat con Gemini", 
                         ui.layout_sidebar(
                             ui.sidebar(
                                 ui.input_text_area("gemini_prompt", "Tu pregunta:", placeholder="Escribe tu pregunta para Gemini aquí..."),
                                 ui.input_action_button("ask_gemini", "Preguntar", icon=ui.tags.i(class_="fas fa-paper-plane"), class_="btn-success"),
                                 width=350
                             ),
                             ui.card(
                                 ui.card_header("Respuesta de Gemini"),
                                 ui.output_text("gemini_response"),
                                 height="400px", style="overflow-y: auto;"
                             )
                         ), 
                         icon=ui.tags.i(class_="fa-solid fa-robot")
            ),
            ui.nav_panel("Mapa (Solo al seleccionar Google Maps)", 
                         ui.output_ui("google_map_embed"), 
                         icon=ui.tags.i(class_="fas fa-map-marked-alt"))
        ),
        icon=ui.tags.i(class_="fas fa-database")
    )

def nav_panel_analisis_y_visualizaciones():
    return ui.nav_panel(
        "Análisis y Visualizaciones",
        ui.navset_card_pill(
            ui.nav_panel("Análisis de Sentimiento", 
                         ui.output_plot("sentiment_output"), 
                         ui.download_button("download_sentiment_plot", "Descargar Gráfico (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"),
                         icon=ui.tags.i(class_="fa-solid fa-magnifying-glass-chart")),
            ui.nav_panel("Análisis de Emociones", 
                         ui.output_plot("emotion_plot_output"), 
                         ui.download_button("download_emotion_plot", "Descargar Gráfico de Emociones (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"),
                         icon=ui.tags.i(class_="fa-solid fa-icons")),
            ui.nav_panel("Análisis de Tópicos", 
                         ui.output_plot("topics_plot_output"), 
                         ui.download_button("download_topics_plot", "Descargar Gráfico de Tópicos (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"),
                         icon=ui.tags.i(class_="fa-solid fa-chart-bar")),
            ui.nav_panel("Mapa Mental", 
                         ui.output_ui("mind_map_output"), 
                         icon=ui.tags.i(class_="fas fa-project-diagram"))
        ),
        icon=ui.tags.i(class_="fas fa-chart-pie")
    )

def nav_panel_comparison_module():
    return ui.nav_panel(
        "Módulo de Comparación",
        ui.markdown("### Comparación de Datos por Sesión"),
        ui.output_text("active_session_id_for_comparison_display"),
        ui.input_action_button("execute_comparison", "Generar Comparación de la Sesión Actual", class_="btn-info mt-2 mb-3"),
        ui.hr(),
        output_widget("comparison_sentiment_plot"),
        ui.hr(),
        output_widget("comparison_emotion_plot"),
        icon=ui.tags.i(class_="fas fa-balance-scale")
    )

def nav_panel_cross_source_comparison():
    return ui.nav_panel(
        "Módulo de Comparación entre redes sociales",
        ui.navset_card_pill(
            ui.nav_panel("Análisis General Comparativo",
                ui.markdown("### Comparación de Datos en la Misma Red"),
                ui.output_text("same_network_active_session_id_display"),
                ui.input_action_button("execute_cross_source_comparison", "Generar/Actualizar Comparación de la Sesión Actual", class_="btn-info mt-2 mb-3"),
                ui.hr(),
                ui.output_ui("comparison_summary_output"),
                ui.hr(),
                output_widget("comparison_sentiment_plot"),
                ui.hr(),
                output_widget("comparison_emotion_plot"),
                ui.hr(),
                output_widget("comparison_topics_plot"),
                icon=ui.tags.i(class_="fas fa-chart-bar")
            ),
            ui.nav_panel("Chatbot Comparativo entre fuentes",
                ui.markdown("### Chatea con los Datos de la Sesión de Comparación"),
                ui.output_text("active_session_id_for_cross_source_comparison_chat_display"),
                ui.hr(),
                ui.input_text_area("comparison_chat_prompt", "Tu pregunta sobre los datos comparados:", rows=3, placeholder="Ej: ¿Cuáles son las principales diferencias entre las fuentes?"),
                ui.input_action_button("ask_comparison_chat", "Preguntar al Chatbot Comparativo", class_="btn-primary mt-2"),
                ui.hr(),
                ui.card(
                    ui.card_header("Respuesta del Chatbot Comparativo"),
                    ui.output_ui("comparison_chat_response_output"),
                    style="min-height: 300px; overflow-y: auto;"
                ),
                icon=ui.tags.i(class_="fas fa-comments")
            ),
            ui.nav_panel("Generador de Infografía (Beta)",
                ui.markdown("### Generar Infografía de la Comparación"),
                ui.input_text("infographic_title", "Título para la Infografía:", placeholder="Ej: Comparativa de Opiniones sobre Producto X"),
                ui.download_button("generate_cross_source_infographic_pdf_handler", "Generar y Descargar Infografía (PDF)", class_="btn-success mt-2 mb-3"),
                ui.output_ui("infographic_generation_status"),
                icon=ui.tags.i(class_="fas fa-file-pdf")
            )
        ),
        icon=ui.tags.i(class_="fas fa-balance-scale")
    )

def nav_panel_comparacion_misma_red():
    return ui.nav_panel(
        "Módulo de Comparación misma red",
        ui.navset_card_pill(
            ui.nav_panel("Análisis General Comparativo",
                ui.output_ui("same_network_comparison_summary_output"),
                ui.input_action_button("execute_same_network_comparison", "Generar Comparación", class_="btn-info mt-2 mb-3"),
                ui.hr(),
                output_widget("same_network_sentiment_plot"),
                ui.hr(),
                output_widget("same_network_emotion_plot"),
                icon=ui.tags.i(class_="fas fa-chart-bar")
            ),
            ui.nav_panel("Chatbot Comparativo Misma Red",
                ui.markdown("### Chatea con los Datos de la Sesión de Comparación"),
                ui.output_text("active_session_id_for_same_network_comparison_chat_display"),
                ui.hr(),
                ui.input_text_area("same_network_comparison_chat_prompt", "Tu pregunta sobre los datos comparados:", rows=3, placeholder="Ej: ¿Cuáles son las principales diferencias entre las fuentes?"),
                ui.input_action_button("ask_same_network_comparison_chat", "Preguntar al Chatbot Comparativo", class_="btn-primary mt-2"),
                ui.hr(),
                ui.card(
                    ui.card_header("Respuesta del Chatbot Comparativo"),
                    ui.output_ui("same_network_comparison_chat_response_output"),
                    style="min-height: 300px; overflow-y: auto;"
                ),
                icon=ui.tags.i(class_="fas fa-comments")
            )
        ),
        icon=ui.tags.i(class_="fa-solid fa-scale-unbalanced-flip")
    )

def nav_panel_scraper_tablas_chatbot():
    return ui.nav_panel(
        "Scrapear Tablas con Chatbot",
        ui.card(
            ui.card_header("Resultados del Scraper y Parser"),
            ui.download_button("download_scraper_parser_table", "Descargar Tabla CSV", icon=ui.tags.i(class_="fas fa-download"), class_="btn-success btn-sm mb-2 mt-2"),
            ui.download_button("download_scraper_parser_table_excel", "Descargar Tabla Excel", icon=ui.tags.i(class_="fas fa-file-excel"), class_="btn-success btn-sm mb-2 mt-2 ms-2"),
            ui.output_ui("scraper_parser_output"),
            style="overflow-y: auto;"
        ),
        icon=ui.tags.i(class_="fa-solid fa-wand-magic-sparkles")
    )