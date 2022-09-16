# Los_Subnormales

Seguimos la normatividad en el ambiente de este reto que era kaggle, por ejemplo al subir solamente el límite de submissions por día.

El link del notebook de google colab donde desarrollamos el modelo:
https://colab.research.google.com/drive/1mrjdmwS8E6C0RwM04qNUW1feNAyGhF0L?usp=sharing

El link del notebook de google colab con la evaluación del modelo:
https://colab.research.google.com/drive/1pndnQEwHYiljE3WZh6FKM4cnaNRYhiro?usp=sharing


## Correcciones

Hicimos correcciones a nuestro proceso de ETL, basándonos en la retroalimentación del profesor Julio y al desempeño que veíamos de nuestro modelo usando la métrica de binary accuracy.

Vimos que dejar los outliers mejoraba el desempeño y no aplicar escalamiento ni normalización mejoraba el desempeño. Además mejoramos nuestro proceso de imputación de datos faltantes al dividir la información por grupos y encontrar relaciones que no seguían la moda o la media que eran los datos que imputábamos inicialmente. Ahondamos sobre estas correcciones en el [notebook](https://colab.research.google.com/drive/1mrjdmwS8E6C0RwM04qNUW1feNAyGhF0L?usp=sharing) donde desarrollamos el modelo.

Con la guía del profesor Víctor, nos enfocamos en los modelos que vimos más en clase como los árboles y la regresión logística, ya que la vimos en ambos módulos de estadística y de aprendizaje.

## Archivos importantes
El servidor desarrollado con Flask que utiliza el modelo y que fue desplegado en una instancia EC2, se encuentra en servidor.py