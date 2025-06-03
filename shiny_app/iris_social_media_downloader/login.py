#logo_path = os.path.join(os.path.dirname(__file__), "www", "LogoNuevo.png")
here = Path(__file__).parent

# Lista de dominios permitidos para el login
DOMINIOS_PERMITIDOS = ["tuempresa.com", "otrodominio.com", "gmail.com"] # ¡¡¡IMPORTANTE: Modifica esto con tus dominios reales!!!


#### Comenzando la UI/Frontend
app_ui = ui.page_fixed(
    ui.tags.head(
            });            
        """)
    ),
    #ui.tags.link(rel="stylesheet", href="styles.css"),
    ui.layout_sidebar(
        ui.sidebar(
            #ui.img(src="LogoNuevo.png", style="height: 40px; width: auto; display: block; margin-left: auto; margin-right: auto; margin-bottom: 10px;"),
            #ui.output_image("app_logo", width='100px', height='50px'),
            #ui.img(src="LogoNuevo.png", height='100px', class_="center-block"),
            #ui.img(src="./www/LogoNuevo.png", height='100px', class_="center-block"),
            #ui.img(src="E:/Users/1167486/Local/scripts/Social_media_comments/shiny_app/iris_social_media_downloader/www/LogoNuevo.png", height='100px', class_="center-block"),            
            ui.markdown("**Social Media Downloader** - Extrae y analiza datos de diferentes plataformas."),
            ui.hr(),
            # Selector de plataforma
            ui.input_select(
                "platform_selector",
                "Seleccionar Plataforma:",
                {
                    "wikipedia": "Wikipedia",
                    "playstore": "Google Play Store",
                    "youtube": "YouTube",
                    "maps": "Google Maps",
                    "reddit": "Reddit", 
                    "twitter": "Twitter (X)",
                    "generic_webpage": "Página web Genérica",
                    "facebook": "Facebook (Próximamente)",
                    "instagram": "Instagram (Próximamente)",
                    "amazon_reviews": "Amazon Reviews (Próximamente)"
                }
            ),
            
            # Inputs dinámicos según plataforma seleccionada
            ui.output_ui("platform_inputs"),
            
            #ui.input_action_button("execute", "Ejecutar", class_="btn-primary"),
            ui.input_action_button("execute", " Scrapear!!", icon=ui.tags.i(class_="fas fa-play"), class_="btn-primary"),
            width=350
        ),
        
        ui.navset_card_tab(
            ui.nav_panel(
                " Base de datos",
                ui.output_data_frame('df_data'),
                #ui.download_button("download_data", "Descargar CSV", class_="btn-info btn-sm mb-2")
                ui.download_button("download_data", "Descargar CSV", icon=ui.tags.i(class_="fas fa-download"), class_="btn-info btn-sm mb-2 mt-2"),
                icon=ui.tags.i(class_="fas fa-table-list")                
                #ui.output_ui("dynamic_content")
                #ui.output_ui('platform_inputs')
            ),
            ui.nav_panel(
                " Resumen",
                #ui.output_text("summary_output"),
                ui.output_ui('styled_summary_output'),
                icon=ui.tags.i(class_="fas fa-file-lines")

            ),
            ui.nav_panel(
                " Análisis de Sentimiento",
                #output_widget("sentiment_output"),
                ui.output_plot("sentiment_output"),
                ui.download_button("download_sentiment_plot", "Descargar Gráfico (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"),                
                #ui.output_ui('styled_summary_output'),
                #icon=ui.tags.i(class_="fa-solid fa-chart-simple")
                #icon=ui.tags.i(class_="fa-solid fa-face-smile"),
                #icon=ui.tags.i(class_="fa-solid fa-face-frown")
                icon=ui.tags.i(class_="fa-solid fa-magnifying-glass-chart")
            ),
            ui.nav_panel(
                " Análisis de Emociones",
                ui.output_plot("emotion_plot_output"),
                ui.download_button("download_emotion_plot", "Descargar Gráfico de Emociones (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"),
                #icon=ui.tags.i(class_="fa-solid fa-face-grin-beam"), 
                #icon=ui.tags.i(class_="fa-solid fa-face-sad-cry"),
                #icon=ui.tags.i(class_="fa-solid fa-face-angry")
                icon=ui.tags.i(class_="fa-solid fa-icons")
            ),
            ui.nav_panel(
                "Map (Solo con Google Maps Selector)",
                ui.output_ui("google_map_embed"),
                icon = ui.tags.i(class_="fas fa-map-marked-alt")
            ),
            ui.nav_panel(
                "Análisis de tópicos",
                ui.output_plot("topics_plot_output"),
                ui.download_button("download_topics_plot", "Descargar Gráfico de Tópicos (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"),
                icon = ui.tags.i(class_="fa-solid fa-chart-bar")
            ),
            ui.nav_panel(
                " Chat con Gemini",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_text_area("gemini_prompt", "Tu pregunta:", 
                                         placeholder="Escribe tu pregunta para Gemini aquí..."),
                        #ui.input_action_button("ask_gemini", "Preguntar", class_="btn-success"),
                        ui.input_action_button("ask_gemini", "Preguntar", icon=ui.tags.i(class_="fas fa-paper-plane"),
                                                class_="btn-success"),
                        width=350
                    ),
                    ui.card(
                        ui.card_header("Respuesta de Gemini"),
                        ui.output_text("gemini_response"),
                        height="400px",
                        style="overflow-y: auto;"
                    )
                ),
                icon=ui.tags.i(class_="fa-solid fa-robot")
            ),
            ui.nav_panel(
                " Mapa Mental", 
                ui.output_ui("mind_map_output"),
                icon=ui.tags.i(class_="fas fa-project-diagram")
            ),   
            ui.nav_panel(
                " Web Scraper/Parser",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_text_area("scraper_urls", "URLs a Scrapear (una sola): \n No admite redes sociales, solo páginas web", 
                                         placeholder="https://ejemplo.com/pagina", value ="https://www.elektra.mx/italika",  height=150),
                        ui.input_text_area("parser_description", "Describe qué información quieres extraer:", 
                                         placeholder="Ej: 'Tabla de precios de productos'", height=100, value = 'Genera una tabla con los precios de las motos de mayor a menor precio'),
                        ui.input_action_button("execute_scraper_parser", "Scrapear y Parsear", 
                                               icon=ui.tags.i(class_="fas fa-play"), class_="btn-primary"),
                        width=350
                    ),
                    ui.card(
                        ui.card_header("Resultados del Scraper y Parser"),
                        #ui.p("Para este caso, no se necesita seleccionar una plataforma del menú de la izquierda."),
                        # This output will dynamically show tables or raw text
                        ui.output_ui("scraper_parser_output"),
                        #height="600px", # Adjust height as needed
                        style="overflow-y: auto;"
                    )
                ),
                icon = ui.tags.i(class_="fa-solid fa-comments")
            )            
        )
    ),
    #theme=shinyswatch.theme.darkly()
    ui.output_ui("ui_dinamica_app"), # Este output renderizará el login o la app principal
    theme=shinyswatch.theme.minty()
)

#### Comenzando el server/Backend
def server(input, output, session):
    # Variables reactivas para la autenticación
    usuario_autenticado = reactive.Value(None) # Almacenará el nombre de usuario (ej: 'nombre.apellido')
    mensaje_login = reactive.Value("")

    # Configuración inicial de las variables para el server
    ## Lazy Load o Carga del Perezoso
    pinecone_client = reactive.Value(None)
    topic_pipeline_instance =  reactive.Value(None)
    map_coordinates = reactive.Value(None)
    mind_map_html = reactive.Value(None)

    # --- Lógica de Autenticación ---
    def ui_login_form():
        """Retorna la UI para el formulario de login."""
        return ui.div(
            ui.row(
                ui.column(4, offset=4,
                    ui.panel_well(
                        ui.h3("Acceso Social Media Downloader", style="text-align: center;"),
                        ui.hr(),
                        ui.input_text("email_login", "Correo Electrónico:", placeholder="usuario@dominio.com"),
                        ui.input_action_button("boton_login", "Ingresar", class_="btn-primary btn-block"),
                        ui.output_text("texto_mensaje_login", style="color: red; margin-top: 10px; text-align: center;")
                    )
                )
            ),
            style="margin-top: 100px;"
        )

    @output
    @render.text
    def texto_mensaje_login():
        return mensaje_login.get()

    @reactive.Effect
    @reactive.event(input.boton_login)
    def manejar_intento_login():
        email = input.email_login()
        if not email:
            mensaje_login.set("Por favor, ingrese su correo electrónico.")
            return

        try:
            nombre_usuario, dominio = email.strip().lower().split('@')
            if dominio in DOMINIOS_PERMITIDOS:
                usuario_autenticado.set(nombre_usuario) # Guardamos solo la parte antes del @
                mensaje_login.set("")
                ui.notification_show(f"¡Bienvenido, {nombre_usuario}!", type="message", duration=5)
            else:
                mensaje_login.set("Dominio de correo no autorizado.")
                usuario_autenticado.set(None)
        except ValueError:
            mensaje_login.set("Formato de correo electrónico inválido.")
            usuario_autenticado.set(None)
        except Exception as e:
            mensaje_login.set(f"Error inesperado durante el login: {e}")
            usuario_autenticado.set(None)

    @reactive.Effect
    @reactive.event(input.boton_logout) # Necesitarás añadir este botón en tu UI principal
    def manejar_logout():
        nombre_usuario_actual = usuario_autenticado.get()
        usuario_autenticado.set(None)
        mensaje_login.set("Sesión cerrada exitosamente.")
        if nombre_usuario_actual:
            ui.notification_show(f"Hasta luego, {nombre_usuario_actual}.", type="message", duration=5)
        else:
            ui.notification_show("Sesión cerrada.", type="message", duration=5)


    # --- UI Principal de la Aplicación (tu UI original) ---
    @reactive.Calc
    def ui_principal_app():
        """Retorna la UI principal de la aplicación cuando el usuario está autenticado."""
        # Aquí va TODA tu UI original que estaba en app_ui = ui.page_fixed(...)
        return ui.layout_sidebar(
            ui.sidebar(
                ui.markdown(f"Usuario: **{usuario_autenticado.get()}**"), # Mostrar usuario logueado
                ui.markdown("**Social Media Downloader** - Extrae y analiza datos de diferentes plataformas."),
                ui.hr(),
                ui.input_select(
                    "platform_selector", "Seleccionar Plataforma:",
                    {
                        "wikipedia": "Wikipedia", "playstore": "Google Play Store", "youtube": "YouTube",
                        "maps": "Google Maps", "reddit": "Reddit", "twitter": "Twitter (X)",
                        "generic_webpage": "Página web Genérica", "facebook": "Facebook (Próximamente)",
                        "instagram": "Instagram (Próximamente)", "amazon_reviews": "Amazon Reviews (Próximamente)"
                    }
                ),
                ui.output_ui("platform_inputs"),
                ui.input_action_button("execute", " Scrapear!!", icon=ui.tags.i(class_="fas fa-play"), class_="btn-primary"),
                ui.hr(),
                ui.input_action_button("boton_logout", "Cerrar Sesión", class_="btn-danger btn-sm btn-block"),
                width=350
            ),
            ui.navset_card_tab(
                ui.nav_panel(" Base de datos", ui.output_data_frame('df_data'), ui.download_button("download_data", "Descargar CSV", icon=ui.tags.i(class_="fas fa-download"), class_="btn-info btn-sm mb-2 mt-2"), icon=ui.tags.i(class_="fas fa-table-list")),
                ui.nav_panel(" Resumen", ui.output_ui('styled_summary_output'), icon=ui.tags.i(class_="fas fa-file-lines")),
                ui.nav_panel(" Análisis de Sentimiento", ui.output_plot("sentiment_output"), ui.download_button("download_sentiment_plot", "Descargar Gráfico (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"), icon=ui.tags.i(class_="fa-solid fa-magnifying-glass-chart")),
                ui.nav_panel(" Análisis de Emociones", ui.output_plot("emotion_plot_output"), ui.download_button("download_emotion_plot", "Descargar Gráfico de Emociones (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"), icon=ui.tags.i(class_="fa-solid fa-icons")),
                ui.nav_panel("Map (Solo con Google Maps Selector)", ui.output_ui("google_map_embed"), icon=ui.tags.i(class_="fas fa-map-marked-alt")),
                ui.nav_panel("Análisis de tópicos", ui.output_plot("topics_plot_output"), ui.download_button("download_topics_plot", "Descargar Gráfico de Tópicos (PNG)", icon=ui.tags.i(class_="fas fa-image"), class_="btn-success btn-sm mt-2"), icon=ui.tags.i(class_="fa-solid fa-chart-bar")),
                ui.nav_panel(" Chat con Gemini", ui.layout_sidebar(ui.sidebar(ui.input_text_area("gemini_prompt", "Tu pregunta:", placeholder="Escribe tu pregunta para Gemini aquí..."), ui.input_action_button("ask_gemini", "Preguntar", icon=ui.tags.i(class_="fas fa-paper-plane"), class_="btn-success"), width=350), ui.card(ui.card_header("Respuesta de Gemini"), ui.output_text("gemini_response"), height="400px", style="overflow-y: auto;")), icon=ui.tags.i(class_="fa-solid fa-robot")),
                ui.nav_panel(" Mapa Mental", ui.output_ui("mind_map_output"), icon=ui.tags.i(class_="fas fa-project-diagram")),
                ui.nav_panel(" Web Scraper/Parser", ui.layout_sidebar(ui.sidebar(ui.input_text_area("scraper_urls", "URLs a Scrapear (una sola): \n No admite redes sociales, solo páginas web", placeholder="https://ejemplo.com/pagina", value="https://www.elektra.mx/italika", height=150), ui.input_text_area("parser_description", "Describe qué información quieres extraer:", placeholder="Ej: 'Tabla de precios de productos'", height=100, value='Genera una tabla con los precios de las motos de mayor a menor precio'), ui.input_action_button("execute_scraper_parser", "Scrapear y Parsear", icon=ui.tags.i(class_="fas fa-play"), class_="btn-primary"), width=350), ui.card(ui.card_header("Resultados del Scraper y Parser"), ui.output_ui("scraper_parser_output"), style="overflow-y: auto;")), icon=ui.tags.i(class_="fa-solid fa-comments"))
            )
        )

    # --- Renderizador Condicional de UI ---
    @output
    @render.ui
    def ui_dinamica_app():
        if usuario_autenticado.get() is None:
            return ui_login_form()
        else:
            return ui_principal_app()

    # --- Resto de tu lógica de servidor ---
    # (Asegúrate de que las funciones que dependen del estado de login
    #  verifiquen `usuario_autenticado.get()` si es necesario)

    
    @render.image
    def app_logo():
    @reactive.Effect
    @reactive.event(input.ask_gemini)
    def ask_gemini_handler():
        # Ejemplo de cómo usar el usuario autenticado:
        current_user = usuario_autenticado.get()
        if not current_user:
            current_gemini_response.set("Error: Debes iniciar sesión para usar esta función.")
            return
        #print('Entró en la función de respuesta de Gemini')
        #genai.configure(api_key=GEMINI_API_KEY)
        #model = genai.GenerativeModel('gemini-1.5-pro')
        if pinecone_context:
            combined_context += f"Contexto relevante de la base de conocimiento:\n{pinecone_context}\n\n---\n\n"
        final_prompt_to_gemini = f"{combined_context}{data_context}Pregunta del usuario: {user_prompt}"
        print(f"Enviando pregunta a Gemini (con contexto si existe):\n{final_prompt_to_gemini[:50]}...") 
        print(f"Usuario '{current_user}' enviando pregunta a Gemini (con contexto si existe):\n{final_prompt_to_gemini[:500]}...") # Log más largo

        try:
            with ui.Progress(min=1, max=3) as p: