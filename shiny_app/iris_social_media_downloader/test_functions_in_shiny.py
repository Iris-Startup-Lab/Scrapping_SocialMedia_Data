import os 
from shiny import App, reactive, render, ui
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import lida 

# En una aplicación real, usarías las funciones de tu script.
# Aquí, la función de búsqueda se implementa con una herramienta ficticia.
def search_api_call(query):
    """Simula una llamada a la API de búsqueda."""
    print(f"Buscando con la consulta: {query}")
    # Retorna un DataFrame de ejemplo. En la realidad, esto sería un resultado de API.
    data = {
        'red_social': ['Twitter', 'LinkedIn', 'Facebook'],
        'enlace': [
            'https://twitter.com/ejemplo_busqueda',
            'https://linkedin.com/in/ejemplo_busqueda',
            'https://facebook.com/groups/ejemplo_busqueda'
        ]
    }
    return pd.DataFrame(data)

# Función para simular la detección de preguntas y generación de gráficos.
def generate_questions_and_plot(content_data):
    """Simula la detección de preguntas y la generación de gráficos con Gemini/Lida."""
    # Aquí iría la lógica para llamar a la API de Gemini para detectar preguntas.
    # Por ejemplo, una llamada a un modelo LLM con un prompt.
    questions = [
        "¿Cuáles son los temas principales de conversación?",
        "¿Cómo es el sentimiento general del contenido?"
    ]

    # Genera datos de ejemplo para el gráfico.
    data = {'Positivo': 50, 'Neutral': 30, 'Negativo': 20}
    labels = list(data.keys())
    sizes = list(data.values())

    # Crea un gráfico de pastel con matplotlib.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107', '#F44336'])
    ax.axis('equal')  # Asegura que el pastel sea un círculo.
    ax.set_title("Sentimiento General del Contenido")
    
    # Guarda el gráfico en una cadena base64.
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return questions, base64_image

app_ui = ui.page_fluid(
    ui.h2("Herramienta de Investigación en Redes Sociales"),
    ui.p("Escribe un prompt de investigación para encontrar información relevante en distintas redes sociales."),
    ui.input_text_area("prompt", "Prompt de Investigación:"),
    ui.input_checkbox_group("redes_sociales", "Selecciona las redes sociales a analizar:", 
                            {"twitter": "Twitter", "facebook": "Facebook", "instagram": "Instagram", "tiktok": "TikTok", "linkedin": "LinkedIn"}),
    ui.input_action_button("buscar_btn", "Generar Búsqueda"),
    ui.hr(),
    ui.output_ui("main_content")
)

def server(input, output, session):
    search_results = reactive.Value(None)
    graphs_data = reactive.Value(None)
    
    @reactive.Effect
    @reactive.event(input.buscar_btn)
    def do_search():
        prompt = input.prompt()
        if not prompt:
            ui.notification_show("Por favor, introduce un prompt de investigación.", duration=3, type="warning")
            return
        
        redes_seleccionadas = input.redes_sociales()
        if not redes_seleccionadas:
            ui.notification_show("Por favor, selecciona al menos una red social.", duration=3, type="warning")
            return
        
        # Simula la generación de palabras clave con DeepSeek
        palabras_clave_extra = "análisis, tendencias, opinión pública" # Marcador de posición
        query = f"{prompt} {palabras_clave_extra} " + " OR ".join([f"site:{red}" for red in redes_seleccionadas])

        with ui.Progress(min=1, max=10) as p:
            p.set(message="Buscando en la web...", detail="Esto puede tomar un momento...")
            df = search_api_call(query) # Llama a la función de búsqueda simulada.
            search_results.set(df)
            p.set(10)

    @render.ui
    def resultados_ui():
        if search_results.get() is None:
            return None
        
        df = search_results.get()
        if df.empty:
            return ui.p("No se encontraron resultados para su búsqueda.")
        
        return ui.div(
            ui.h3("Resultados de la Búsqueda"),
            ui.output_table("tabla_resultados"),
            ui.input_action_button("confirmar_btn", "Confirmar y Analizar Links")
        )

    @render.table
    def tabla_resultados():
        return search_results.get()

    @reactive.Effect
    @reactive.event(input.confirmar_btn)
    def do_analysis():
        df = search_results.get()
        if df is None or df.empty:
            ui.notification_show("No hay links para analizar.", duration=3, type="warning")
            return

        with ui.Progress(min=1, max=10) as p:
            p.set(message="Extrayendo información y generando gráficos...", detail="Esto puede tomar un momento...")
            
            # Simular la extracción de contenido y el análisis.
            content = "Contenido extraído de los links." # Marcador de posición.
            
            # Llamada a la función simulada de análisis y visualización.
            questions, base64_image = generate_questions_and_plot(content)
            
            graphs_data.set({'questions': questions, 'image': base64_image})
            p.set(10)
    
    @render.ui
    def graphs_ui():
        data = graphs_data.get()
        if data is None:
            return None
        
        questions = data['questions']
        image_base64 = data['image']
        
        # Muestra las preguntas detectadas.
        questions_list = [ui.tags.li(q) for q in questions]
        
        return ui.div(
            ui.h3("Análisis de Contenido"),
            ui.h4("Preguntas detectadas por Gemini:"),
            ui.tags.ul(*questions_list),
            ui.h4("Gráficos generados por Lida:"),
            ui.tags.img(src=f"data:image/png;base64,{image_base64}")
        )
    
    @render.ui
    def main_content():
        if graphs_data.get() is None:
            return ui.output_ui("resultados_ui")
        else:
            return ui.output_ui("graphs_ui")

app = App(app_ui, server)


if __name__ == "__main__":
    app.run()


"""
import os
from shiny import App, ui, render, reactive
import pandas as pd
#from lida import Manager, llm
import lida 
import llmx
import matplotlib.pyplot as plt
import seaborn as sns 
import tempfile
from dotenv import load_dotenv

from accelerate import disk_offload

from transformers import BitsAndBytesConfig
import torch

print('Comenzando el script')
# Configuración de cuantización para reducir uso de memoria

#model ='distilbert/distilgpt2'
#model = 'o4-mini'
#model = 'HuggingFaceH4/zephyr-7b-beta'
model = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
load_dotenv(override=True)
#disk_offload(model=model, offload_dir="offload")
# Inicializar LIDA con Gemini (requiere API key de Google configurada)
#lida = Manager(text_gen=llm("gemini-2.5-flash-lite", ))
text_gen_config = lida.TextGenerationConfig(
    model= model
)

text_gen = lida.llm(
    provider='hf',
    api_key=os.getenv('HUGGING_FACE_API_KEY'),
    device_map='cpu',
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,  
    offload_folder="E:/Users/1167486/Local/offload",  # Carpeta para offloading 
    temperature=0, 
    max_tokens=650, 
    use_cache=False
)


#text_gen = lida.llm(
#    provider='openai',
#    api_key=os.getenv('OPEN_AI_API_KEY')


lida  = lida.Manager(text_gen = text_gen)
##3 Configurando la extensión Geminosa

# Interfaz de usuario
app_ui = ui.page_fluid(
    ui.h2("Visualización automática con LIDA y Gemini"),
    ui.input_file("archivo", "Sube un archivo CSV", accept=".csv"),
    ui.output_plot("grafico")
)

# Servidor
def server(input, output, session):

    @reactive.Calc
    def datos():
        archivo = input.archivo()
        if archivo is None:
            return None
        info_archivo = archivo[0]
        df = pd.read_csv(info_archivo["datapath"])
        return df

    @output
    @render.plot
    def grafico():
        df = datos()
        if df is None:
            return
        local_scope = {'data': df}
    

        # Guardar temporalmente el archivo CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df.to_csv(tmp.name, index=False)
            summary = lida.summarize(tmp.name, textgen_config=text_gen_config)
            goals = lida.goals(summary, n=2)
            #charts = lida.visualize(summary=summary, goal=goals[1], library="matplotlib")
            charts = lida.visualize(summary=summary, goal=goals[0], library="seaborn")
        # Ejecutar el código generado por LIDA
        exec(charts[0].code, globals(), local_scope)

# Crear la app
app = App(app_ui, server)


if __name__ == "__main__":
    app.run()
"""