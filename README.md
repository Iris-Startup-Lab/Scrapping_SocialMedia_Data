# Iris Social Media Downloader

El objetivo de este repositorio es mostrar los métodos para obtener los 
comentarios de diferentes redes sociales y obtener lo siguiente:

- Nombre de usuario
- Comentario
- Calificación (Si aplica)
- Likes o interacciones (Si aplica)
- Fecha de publicación
- Número de respuestas (Si aplica)




## Redes Sociales objetivo

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
Los tiempos de desarrollo aún no se han definido, pero esto se evaluará en conjunto con las demás personas integrantes del equipo de Iris.

