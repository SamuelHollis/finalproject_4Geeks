# Web Scraping y Natural Language Processing: 

El proyecto busca sondear el resultado de las elecciones de EE.UU. del 5 próximo 5 de noviembre mediante la recopilación de información de las principales páginas de debate político en Reddit, usando la técnica de Web Scraping.

Tras un filtrado, utilizaremos modelos de procesamiento de lenguaje natural (NLP), concretamente "sentiment analysis" para evaluar la inclinación política en los datos recopilados.

## Instalación: pasos para instalar y configurar el proyecto:

El repositorio está organizado de la siguiente manera:


- `all_the_news` - EDA y unsupervised learning model para clusters de espectros políticos. 
- `modeling` - RoBERTa model y el análisis de sentimiento.
- `reddit_scraping` - Web Scraping y datasets obtenidos para el análisis de sentimiento.
- `requirements.txt` - librerías necesarias para el desarrollo del proyecto.

## Instalación

Las librerías necesarias para poder desarrollar este proyecto se encuentran disponibles en el archivo 'requirements.txt'. 

Es importante desarrollar este proyecto con Python v.3.8.19 para que la librería Torch pueda funcionar correctamente. 

```bash
pip install -r requirements.txt
```


## Analiza con tu propio dataset: ¡sondea el resultado electoral del próximo 5 de noviembre tú mismo!

Gracias a la función scrapeadora de la carpeta ´reddit_scraping´ podrás obtener la recopilación de comentarios más recientes en un momento concreto. 

Según los expertos en política, los resultados de unas elecciones se juegan, sobre todo, en las tres semanas previas al día de la votación. 

Nuestra recomendación es practicar el Web Scraping en este periodo, para, fiinalmente, analizarlos con el modelo RoBERTa. 
