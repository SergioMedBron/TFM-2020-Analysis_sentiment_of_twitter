# TFM-2020-Analysis_sentiment_of_twitter

## Resumen:

En este repositorio encontraras dos archivos realizados en Jupyter notebook que tienen como finalidad la aplicabilidad práctica del análisis de sentimiento político  sobre el  texto.
En el primer archivo [nombre] encontrarás la construcción de una red neuronal convolucional. 
En el segundo la extracción, aplicación del modelo y la posterior información gráfica obtenida de ello.

El análisis se ha realizado sobre la red social Twitter por la facilidad de la obtención de los datos mediante la api y el coste nulo de acceso a ella


## Introducción:

La transformación digital ha cambiado el paradigma empresarial, siendo de vital importancia el contacto con los clientes, llegando a denominarse esta filosofía empresarial como "Costumer Centric" o centrado en el usuario.
Por ello han aparecido puestos de trabajo cuya finalidad es crear la imagen de la empresa o la marca en las distintas redes sociales y obtener información directa de los usuarios, siendo esta presencia especialmente relevante a la hora de atender las necesidades de los usuarios de sus productos o servicios.

A su vez la necesidad derivada del entorno de alta incertidumbre en el que nos encontramos, debido a la aparición de nuevas empresas que ofrecen servicios disruptivos a los usuarios que han llevado a la crisis a numerosas empresas que antaño eran líderes en su sector o la disrupción de nuevos modelos de negocio que hacen tambalear sectores enteros como la aparición de AIRBNB sobre el sector hotelero, lleva a las empresas a la necesidad de innovar constantemente en sus productos.
Estas innovaciones surgen de la una aplicación de metodologías ya probadas y eficaces en todo el mundo basadas en construir los productos y servicios nuevos entorno a las necesidades los usuarios.

Es por ello que el análisis de las opiniones de los usuarios es vital para las empresas y siendo esto sabido, durante el año 2019 surgieron multitud de aproximaciones al estudio de esta necesidad.

El enfoque de este trabajo nace de la misma premisa pero se centra en el ámbito político, el objetivo es analizar la dispersión del sentimiento de un grupo específico y poder obtener la capacidad de influencia de los sujetos estudiados.


## Tecnología:

Todo el proyecto está realizado en python y elaborado utilizando el ide Jupyter Notebooks, sin embargo el proyecto está pensado para poder ejecutarse automáticamente añadiendo una función que recoja los datos de entrada necesarios

Todas las librerías necesarias para la ejecución se pueden encontrar en el archivo requeriments.txt y ejecutar su instalación con pip 

Para la elaboración de la red neuronal se utilizó un embedding pre-entrenado descargado de https://github.com/dccuchile/spanish-word-embeddings y que se puede obtener del siguiente drive https://drive.google.com/drive/folders/1nKmFegE0SzxKQeLKEZ-SVZXAmerCt8_Z?usp=sharing
Para poder ejecutar el archivo y obtener la visualización es necesario descargarse el modelo  del enlace al drive anterior  llamado Keras_model_with_stopwords_067_acc.sav

Nota:1.1 El acceso a la api de twitter es gratuito pero está limitado a un número de peticiones obligándote a esperar 15 min cuando estas son alcanzadas, por lo que se recomienda no utilizar como sujeto de pruebas,  una cuenta con número muy elevado de seguidores, en el drive se encuentra también una carpeta denominada prueba, con los archivos necesarios para la ejecución sin necesidad de realizar ninguna llamada a la api acortando los tiempos de ejecución, dentro de la carpeta se encuentra un archivo denominado instrucciones donde se explica cómo hacer para ejecutar el proceso.      
1.2 Debido al tamaño del modelo, se recomienda ejecutar el proyecto con un mínimo de 8gb de RAM, ya que mientras se ejecute, en función de la capacidad del procesador, el modelo utilizará un mínimo de 5gb de RAM.
	
## Estado del arte:
### NLP 
Hasta la fecha los mayores descubrimientos realizados en el ámbito del data science están centrados en el procesamiento de imágenes, donde durante los últimos años se ha experimentado un crecimiento exponencial en la aparición de soluciones.

Exceptuando la empresa Google, la cual ha sido la precursora en muchos aspectos de esta tecnología, implementándola en la mayoría de sus líneas de negocio, desde la detección de spam, hasta la indexación de las páginas webs, a comienzos de la realización de este trabajo, el campo del procesamiento natural del lenguaje no tenía tanta importancia para la comunidad, siendo el modelo GPT-2 creado por OpenAI el principal estandarte de la tecnología

A finales de 2019 y principios de 2020 hemos experimentado la misma evolución que ya tuvimos en el procesamiento de imágenes, en este campo con la presentación del modelo GPT-3 y el chat de Facebook Blender-bot como principales ejemplos de esto.

Analizando las tecnologías presentadas, esta evolución parece haberse estancado en una arquitectura que ha demostrado ser muy efectiva, denominada "transformer" y basada en la arquitectura de red LSTM 
Esta arquitectura fue presentada por Google y es una técnica de aprendizaje no supervisado cuya finalidad consiste en predecir la siguiente palabra dado un input cualquiera.

Este tipo de arquitectura ha dado grandes resultados para problemas no planteados anteriormente, como la capacidad de resumir un texto o traducir de un idioma a otro.

Los avances registrados en el último año sobre NLP han consistido en aumentar los parámetros de entrenamiento para esta arquitectura, dando como resultado modelos imposibles de entrenar por el hardware "domestico".


### Análisis de sentimiento:
Como se menciona anteriormente, el año 2019 fue un año muy fructífero para el análisis de sentimiento, y se generaron multitud de respuestas a este fenómeno donde se combinaban distintas tecnologías siendo incluso uno de los artículos con mayor influencia en Medium (servicio de publicación de blogs profesionales) un artículo de recopilación científica escrito por Lei Zhang, Shuai Wang, Bing Liu denominado Deep Learning for Sentiment Analysis : A Survey, donde se recogen las principales herramientas utilizadas por la comunidad para la solucionar el problema planteado.
El artículo se resume en la siguiente tabla, donde se muestran las principales tecnologías utilizadas por la comunidad y los principales inconvenientes derivados de ella.
 


Si hacemos una búsqueda en kaggle existen multitud de repositorios que utilizan distintas arquitecturas para resolver el problema llegando en la mayoría de ocasiones a un 80% de precisión en sus predicciones.

Así mismo, en python existe una librería llamada Textblob que cumple esta finalidad, el problema de esta librería es que esta entrenada en inglés y utiliza la api de google translator para generar las predicciones en español limitando el número de predicciones al límite gratuito proporcionado por Google.


## Arquitectura utilizada:

La ejecución de este proyecto está más centrada en la información visual que el análisis del sentimiento pueda proporcionar, es decir, el objetivo es analizar la capacidad de influencia de un agente dentro de su red, comprobar la polarización sentimental de los sujetos.

Por ello se usa como arquitectura una red convolucional (CNN) con un Word embbeding pre-entrenado, enteramente en español, al que se le darán como datos de entrenamiento un conjunto de tweets, previamente etiquetados de forma manual obtenidos de http://tass.sepln.org/2017/#datasets.

**El flujo de trabajo de la arquitectura del script completo consiste en la entrada de forma manual del id de twitter del sujeto de estudio y una fecha concreta, en formato YY/MM/DD.**

Una vez ejecutamos el archivo, se descargaran (en la propia carpeta donde se encuentre el ejecutado), los id de los seguidores y los tweets que haya realizado durante ese día y se creara dos archivo con la información obtenida.

Una vez finalizado este proceso se iniciará el mismo proceso con cada uno de los seguidores del sujeto de estudio.

Tras obtener los archivos se iniciaran la limpieza y procesado de los tweets para generar las predicciones de los tweets que se guardaran en forma de diccionario en archivos

Una vez realizadas las predicciones, se procederá a la lectura de los diccionarios y a la posterior visualización de los mismos, creando un archivo HTML que permite la interacción con los nodos, así como aumentar o disminuir el zoom para su mejorar su análisis.

*La decisión de crear los archivos de esta forma es reducir el coste de la memoria RAM y la recuperación del proceso en caso de fallo del sistema que está ejecutando el proceso debido a la demora provocada por las condiciones de la api*


## Entrenamiento del modelo:

En modelo se basa en una red convolucional creada mediante la librería Keras.

La elección de esta arquitectura se basa en la necesidad de obtener una clasificación final dada en un intervalo entre 0 y 1, siendo 0 negativo y 1 positivo.

Pese al déficit producido por el tamaño del data set de entrenamiento, se ha decidido eliminar del proceso de entrenamiento aquellos tweets etiquetados como neutrales  por considerar que de estar, estos solo generarían ruido y bajaría la calidad de las predicciones.

### Word Embbeding o Layer Embbeding:

En la construcción del modelo presentado se tomó la decisión de usar un Words Embbeding ya entrenado con el objetivo de reducir el coste computacional del entrenamiento y minimizar el error decisión basada en el siguiente artículo, el cual muestra que no hay diferencias significativas en el uso de una u otra para la solución del problema propuesto

### Pre procesamiento:
A la hora del pre procesamiento de los datos, existen tres procesos comúnmente aceptados en los proyectos de NLP, la eliminación de stopwords, la lemmatizacion y el stemming,

### Stopwords (si o no):
Las denominadas stopwords son palabras que tiene mucha recurrencia en el lenguaje y que a priori no afectan al mensaje comunicado, dadas las discrepancias existentes en la comunidad a cerca de este proceso,  se entrenaron dos modelos y la decisión fue tomada en torno a la precisión de estos.
Siendo el modelo en el que no se le eliminaron las stopwords aquel que arrojo mejores resultados

### Lemmatizacion y stemming:
Estos dos procesos consisten en reducir la palabra a su raíz, de forma que se eliminan los tiempos verbales o las referencias a los plurales etc...
En el proceso de entrenamiento de esta red, al utilizar un word embbeding, no se ha realizado este proceso ya que el corpus del Embbedings es lo suficientemente grande (1.400 millones de palabras) como para que este proceso supusiera algún resultado y así se disminuye el coste computacional.



## Decisiones de producción y Problemas del modelo:

A la hora de realizar el proyecto se enfentaron los siguientes problemas:

El limitado tamaño de los datos de entrenamiento, ya que el lenguaje utilizado es el español, apenas hay recursos etiquetados para la clasificación de este problema,  en un principio se optó por utilizar un data set en inglés y traducir los resultados, pero las principales Apis de traducción son de pago, lo que dispararía los coste de este proyecto
Por lo que se decidió utilizar los datos existentes pese al detrimento de la precisión del modelo

La adaptabilidad del lenguaje al medio, y es que twitter es una red social cuyas interacciones son diarias, los que implica que los usuarios van adaptando el lenguaje que usan en función de las interacciones lo que genera nuevas palabras y la re significación de algunas de ellas, como por ejemplo en determinados colectivos provenientes del "mundo gamer" la letra "F" se convirtió en una forma de expresar la tristeza.
Esto provoca que las redes neuronales que realicen predicciones sobre esta letra, carezcan de esa nueva categoría, y fallaran en las predicciones.

La polarización del sentimiento a negativo o positivo:
Existen 5 emociones de las cuales se derivan los sentimientos, partir de una digitalización del sentimiento a 0 o 1 hace que se genere mucho ruido en la distinción de los datos etiquetados, ya que si se tiene un comentario triste se atribuirá a uno negativo cuando no tiene por qué significar eso.

Los emoticonos son la herramienta que utilizamos las personas para dotar de emoción a los tweets pero la falta de consenso en su significado y la digitalización de este tipo de estudios (anteriormente mencionada) hace que sea muy complicado encontrar datos etiquetados para el entrenamiento.


## Futuras mejoras:

Pese a los principales problemas anteriormente mencionados, el presente proyecto tiene un gran potencial actual, tanto como de mejora.

Los principales puntos a mejorar son principalmente:

-Aumento del accuracy, ya sea mediante el aumento de los datos de entrenamiento o mediante la ejecución de arquitecturas paralelas que triangulen las predicciones.

-Mejora de la interfaz gráfica y el añadido de las principales palabras usadas por la comunidad estudiada


