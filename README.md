# Reto CONABIO
Este proyecto consiste en entrenar un modelo supervisado para clasificar los pixeles de una imagen satelital.
(No se incluyen los archivos GeoTiff dentro del proyecto, pues son más grandes que lo que Git acepta)

## Modelos
Los modelos que se escogieron fueron Naive Bayes y Máquinas de Soporte Vectoriales lineales. Cada uno cuenta con sus ventajas y desventajas

### Naive Bayes
Este es un modelo muy simple basado en el teorema de Bayes y usando una estimación Maximum a posteriori.
#### Ventajas
* Es un modelo rápido de entrenar
* No tiene mucho problema en clasificar clases con poca representatividad, puesto que se fundamenta en la probabilidad de que aparezca una observación
#### Desventajas
* Si nunca se ha observado una clase, inmediatamente se le clasificará con un cero de probabilidad
* Se asume que la distribución de los datos es gaussiana. Esta es una suposición importante que no siempre se va a cumplir

### Máquina de Soporte Vectorial
Este modelo intenta encontrar hiperplanos que correctamente dividan al conjunto de datos con un cierto margen C.
#### Ventajas
* Su entrenamiento no necesita suposiciones, tan solo ajustar los hiperplanos, por lo que lo hace una herramienta más poderosa que Naive Bayes.
#### Desventajas
* El algoritmo de entrenamiento tiene una complejidad cúbica, por lo que puede tardar horas en entrenarse (Para este proyecto, el máximo número de datos con el que fue entrenado fue un millón)
* Todas las clases son tratadas de la misma manera, por lo que se puede sesgar el modelo al asumir que todas las clases tienen la misma representatividad. Esto se puede arreglar asignando pesos a cada clase.

## Resultados
### Naive Bayes
5 pliegues: 0.5310721123732605, 0.3437910310758086, 0.351717063412779, 0.2348400675150955, 0.06293181981871214
Mean model accuracy 0.304870

### Naive Bayes con NDVI 
5 pliegues: 0.4227434587980633, 0.30324294856901085, 0.2660456085504656, 0.39476471724512857, 0.06855807777857134
Mean model accuracy: 0.291071

### SVM 
5 pliegues: 0.19217796571478854, 0.21371071149712403, 0.17943348072501883, 0.5041955852408216, 0.5781976024968893
Mean model accuracy: 0.333543

### SVM con NDVI
5 pliegues: 0.5408095753049803, 0.36301538496773567, 0.32941906099050466, 0.30880381295751014, 0.700589054442107
Mean model accuracy: 0.448527

(Nota: Es probable que los modelos de SVM estén sesgados en tanto que por cuestiones de tiempo solo se entrenó un un subconjunto del total datos disponibles)

## Conclusiones
Claramente los modelos elegidos no son los mejores para la tarea en cuestión. 
En general cuando se hace clasificación de imágenes, resulta insuficiente tomar como característica los pixeles crudos. En el contexto de clasificación de imágenes, suele ser útil la información que los vecinos proporcionan, de manera que se pueda realizar un contraste. 
Si se buscara un modelo mucho más complejo, el estado del arte en cuanto a clasificación de imágenes sugeriría que se entrenara una red neuronal convolucional. Esta debería entrenarse de manera que aceptara los cuatro canales iniciales, y se debería segmentar la imagen en partes y entrenarla poco a poco para no tener problemas de memoria.
Adicionalmente, se podría utilizar otro modelo (aunque no supervisado) para limpiar o mejorar las clasificaciones, estos son los campos aleatorios condicionales.



