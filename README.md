# Iris Social Media Downloader

El objetivo de este repositorio es mostrar los métodos para obtener los 
comentarios de diferentes redes sociales y obtener lo siguiente:

- Nombre de usuario
- Comentario
- Calificación (Si aplica)
- Likes o interacciones (Si aplica)
- Fecha de publicación
- Número de respuestas (Si aplica)

## Software a utilizar
### Sitios para alojar bases de datos y aplicación
#### Firebase
Este sitio ([Firebase](https://firebase.google.com))  nos ayudará a alojar la aplicación. 


#### SupaBase
Este sitio ([Supabase](https://supabase.com/)) nos permite crear una base de datos gratuita con 500 Mb de almacenamiento.


### FrontEnd
El frontend se planea realizarlo con [Vue.js](https://vuejs.org/) o [Angular.js](https://angular.dev/) usando el lenguaje de programación Javascript (Typescript) con [Node](https://nodejs.org)

### BackEnd
El Backend se realizará con [Django](https://www.djangoproject.com/) usando el lenguaje de programación Python.


### Análisis de sentimientos y emociones
El análisis de sentimientos involucrar la librería [Spacy](https://spacy.io/), así como modelos preentrenados de HuggingFace o 
desarrollar modelos propios con la librería [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert).
Esto con el lenguaje de programación Python.

### Gráficos
Los gráficos se realizarán con diversas librerías, principalmente [MatPlotLib](https://matplotlib.org/) y [Plotly](https://plotly.com/) 

## Redes Sociales objetivo
### Cualquier web 
Para el caso de cualquier sitio web. 
Se deberá generar un proceso generalista donde se obtenga el texto.
Se generen las frases más importantes, así como un resumen, se genere un análisis de sentimientos.
Esto se guardará en base de datos como 2 factores: Frase y sentimiento, añadiendo la url del sitio web 

### Twitter (X)
Para el caso de Twitter, se ha logrado tener acceso a través de 
[X Developers](https://developer.x.com/en) con ayuda de la librería
[Tweepy](https://www.tweepy.org/)
Pero estamos limitados a 100 tweets mensuales (respuestas incluídas) y la versión
de pago cuesta $200 USD por lo que se buscarán alternativas usando Web Scrapping.

El método alternativo comprende utilizar 
[BeautifulSoup](https://pypi.org/project/beautifulsoup4/) junto con [Selenium](https://www.selenium.dev/)

Si esto no llegara a funcionar, usaremos LLLMs con 
[Ollama](https://ollama.com/) junto con [Cursor](https://www.anthropic.com/) para poder hacerlo de manera más efectiva.





### YouTube
Google nos ha aprobado hasta 10 mil request diarios para poder 
obtener comentarios de cualquier video público.
[Google for Developers](https://console.developers.google.com/)


### Facebook
Para el caso de Facebook, se está esperando a la aprobación de la app 
para poder usar sus endpoint
[Facebook Developers](https://developers.facebook.com)

Si la aplicación no es aprobada, se evualuarán métodos alternativos como los que se usarán en Twitter.

### Amazon Comments
Se está desarrollando también un script para poder minar los comentarios de los reviews de productos de Amazon

## Objetivo Final
El objetivo final es crear una aplicación interactiva que con solo un link 
haga el scrapping de todas estas redes sociales anteriormente mencionadas.
 


## Tiempos
Usted puede ver los cambios en este [Timeline](./Timeline.md) que hemos creado para una mejor comprensión de usted de los tiempos de ejecución de esta aplicación

## Desarrolladores:
Iris Startup Lab
Equipo Data & Analytics
Fernando Dorantes Nieto