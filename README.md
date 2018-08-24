# Reto CONABIO
Este proyecto consiste en entrenar un modelo supervisado para clasificar los pixeles de una imagen satelital.
(No se incluyen los archivos GeoTiff dentro del proyecto, pues son más grandes que lo que Git acepta)

##Modelos
Los modelos que se escogieron fueron Naive Bayes y Máquinas de Soporte Vectoriales lineales. Cada uno cuenta con sus ventajas y desventajs

###Naive Bayes
Este es un modelo muy simple basado en el teorema de Bayes y usando una estimación Maximum a posteriori.
####Ventajas
* Es un modelo rápido de entrenar
* No tiene mucho problema en clasificar clases con poca representatividad, puesto que se fundamenta en la probabilidad de que aparezca una observación
#### Desventajas
* Si nunca se ha observado una clase, inmediatamente se le clasificará con un cero de probabilidad
* Se asume que la distribución de los datos es gaussiana. Esta es una suposición importante que no siempre se va a cumplir

###Máquina de Soporte Vectorial
Este modelo intenta encontrar hiperplanos que correctamente dividan al conjunto de datos con un cierto margen C.
####Ventajas
* Su entrenamiento no necesita suposiciones, tan solo ajustar los hiperplanos, por lo que lo hace una herramienta más poderosa que Naive Bayes.
#### Desventajas
* El algoritmo de entrenamiento tiene una complejidad cúbica, por lo que puede tardar horas en entrenarse (Para este proyecto, el máximo número de datos con el que fue entrenado fue un millón)
* Todas las clases son tratadas de la misma manera, por lo que se puede sesgar el modelo al asumir que todas las clases tienen la misma representatividad. Esto se puede arreglar asignando pesos a cada clase.

