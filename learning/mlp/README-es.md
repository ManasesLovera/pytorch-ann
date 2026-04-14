# MLP: Perceptrón Multicapa (Multi-Layer Perceptron)

## Conceptos clave iniciales

Antes de aprender sobre los MLPs, es útil entender algunas ideas centrales que aparecen en casi todas las redes neuronales.

## ¿Qué es una red neuronal?

Una red neuronal es un modelo que aprende patrones ajustando muchos parámetros numéricos llamados **pesos** (weights).

En lugar de escribir reglas explícitas a mano, le das al modelo ejemplos:

- entradas (inputs)
- salidas esperadas (expected outputs)

Durante el entrenamiento, el modelo cambia sus pesos internos para que sus predicciones se acerquen más a las respuestas correctas.

A un nivel muy alto, una red neuronal es solo una pila de funciones matemáticas que transforman datos de entrada en una salida.

Ejemplo:

- entrada: edad, salario y puntaje crediticio de una persona
- salida: probabilidad de aprobación de un préstamo

La red aprende cómo se relacionan esas entradas con la salida al ver muchos ejemplos.

## ¿En qué se diferencia una red neuronal de los modelos de machine learning clásicos?

Tanto el machine learning clásico como las redes neuronales aprenden de los datos, pero suelen diferir en cuánta "extracción de características" (feature learning) hacen por su cuenta.

Los modelos tradicionales a menudo incluyen:

- regresión lineal
- regresión logística
- árboles de decisión
- random forests
- gradient boosting

Estos modelos suelen funcionar muy bien en datos tabulares estructurados, especialmente cuando las características ya son útiles por sí mismas.

Las redes neuronales se diferencian porque pueden aprender **representaciones intermedias** dentro de capas ocultas en lugar de depender solo de características diseñadas manualmente.

En términos simples:

- el ML clásico a menudo depende más de la ingeniería de características (feature engineering).
- las redes neuronales a menudo aprenden características internas útiles de forma automática.

Dicho esto, las redes neuronales no son automáticamente mejores. Para datos tabulares, los modelos de ML clásicos suelen ser competitivos o incluso más fuertes que un MLP, especialmente en conjuntos de datos pequeños.

## ¿Qué es una característica (feature)?

Una **característica** es una variable de entrada utilizada por el modelo.

Ejemplos:

- edad
- salario
- número de compras anteriores
- intensidad de un píxel
- valor de un "embedding" de palabra

Si tu conjunto de datos tiene 10 columnas usadas como entradas, entonces tu modelo tiene 10 características.

## ¿Qué es una neurona?

Una neurona es una pequeña unidad de computación dentro de la red.

Esta unidad:

1. recibe valores de entrada
2. los multiplica por pesos
3. los suma junto con un sesgo (bias)
4. aplica una función de activación
5. pasa el resultado hacia adelante

Esto está inspirado vagamente en la biología, pero en machine learning es simplemente matemática.

## ¿Qué es una capa (layer)?

Una capa es un grupo de neuronas que operan en la misma etapa.

Los tipos de capas comunes en un MLP son:

- **capa de entrada** (input layer): recibe las características.
- **capas ocultas** (hidden layers): transforman la información.
- **capa de salida** (output layer): produce la predicción final.

Cuando la gente dice que un modelo es "profundo" (deep), generalmente se refieren a que tiene muchas capas ocultas.

## ¿Qué significa "hacia adelante" (feedforward)?

**Feedforward** significa que los datos se mueven en una sola dirección:

```text
entrada -> capas ocultas -> salida
```

No hay bucles a través del tiempo ni memoria interna de pasos anteriores.

Esto es diferente de los modelos recurrentes como RNNs y LSTMs, donde los pasos anteriores de una secuencia pueden afectar a los posteriores a través de un estado oculto.

Los MLPs son redes feedforward.

## ¿Qué es una función de activación?

Una función de activación es la parte no lineal de una neurona.

Ejemplos comunes:

- `ReLU`
- `Sigmoid`
- `Tanh`

Sin funciones de activación, apilar múltiples capas se colapsaría en algo similar a una sola transformación lineal. Eso haría que la red fuera mucho menos expresiva.

La función de activación es lo que ayuda a una red neuronal a aprender patrones más complejos.

## ¿Qué es una predicción?

Una predicción es simplemente la salida del modelo para una entrada dada.

Ejemplos:

- una probabilidad de la clase 1
- uno de varios puntajes de clase
- el precio de una casa
- un valor numérico futuro

## ¿Qué es una función de pérdida (loss function)?

Una función de pérdida mide qué tan equivocadas están las predicciones del modelo.

Ejemplos:

- `BCELoss` para clasificación binaria.
- `CrossEntropyLoss` para clasificación multiclase.
- `MSELoss` para regresión.

El entrenamiento intenta reducir esta pérdida con el tiempo.

## ¿Qué es la retropropagación (backpropagation)?

La retropropagación es el método utilizado para calcular cómo contribuyó cada peso al error.

Después de que el modelo hace una predicción y se calcula la pérdida:

1. PyTorch calcula los gradientes.
2. cada gradiente te dice cómo debería cambiar un peso para reducir la pérdida.
3. el optimizador usa esos gradientes para actualizar los pesos.

Normalmente no escribes la retropropagación a mano en PyTorch. Al llamar a:

```python
loss.backward()
```

le pides a PyTorch que calcule esos gradientes automáticamente.

## ¿Qué es un gradiente?

Un gradiente te dice qué tan sensible es la pérdida a un parámetro.

Si un pequeño cambio en un peso hace que la pérdida cambie mucho, ese peso tiene un gradiente grande.

Los gradientes son las señales utilizadas para mejorar el modelo durante el entrenamiento.

## ¿Qué es un optimizador?

Un optimizador actualiza los pesos del modelo después de que se calculan los gradientes.

Opciones comunes:

- `SGD`
- `Adam`

El optimizador es la parte que realmente cambia los pesos de un paso de entrenamiento al siguiente.

## ¿Qué es un "epoch" y un "batch"?

Estos términos aparecen constantemente en deep learning:

- **batch**: un pequeño grupo de ejemplos de entrenamiento procesados juntos.
- **epoch**: una pasada completa por todo el conjunto de datos de entrenamiento.

Ejemplo:

- tamaño del dataset: 1000 filas
- tamaño del batch: 100
- batches por epoch: 10

Entrenar durante 20 epochs significa que el modelo ve todo el conjunto de datos 20 veces.

## ¿Por qué usar batches en lugar de todo el dataset a la vez?

Los batches se usan porque:

- caben mejor en la memoria.
- hacen que el entrenamiento en GPU sea práctico.
- a menudo ayudan a que la optimización se comporte mejor.

## ¿Por qué importa CUDA aquí?

PyTorch puede entrenar redes neuronales ya sea en:

- CPU
- GPU

Para muchas cargas de trabajo de redes neuronales, una GPU es mucho más rápida porque puede procesar muchas operaciones numéricas en paralelo.

Es por eso que mover tensores y modelos a `cuda` es tan común en el código de PyTorch.

## ¿Qué es un MLP?

Un MLP, o Perceptrón Multicapa, es una de las arquitecturas de redes neuronales más básicas.

Es una **red neuronal feedforward**, lo que significa que la información se mueve desde la capa de entrada, a través de una o más capas ocultas, hasta la capa de salida. No retrocede en el tiempo como una RNN, y no escanea regiones locales de imagen como una CNN.

Un MLP suele estar construido con **capas totalmente conectadas** (fully connected layers). Eso significa que cada neurona en una capa se conecta con cada neurona en la capa siguiente.

En la práctica, los MLPs se usan comúnmente para:

- datos tabulares
- clasificación básica
- regresión
- conjuntos de datos estructurados pequeños

## ¿Por qué se llama Perceptrón Multicapa?

El nombre tiene tres partes:

- **Perceptrón**: el modelo original simple similar a una neurona que toma entradas, aplica pesos, suma un sesgo y produce una salida.
- **Capa** (Layer): las neuronas se agrupan en capas.
- **Multicapa**: en lugar de solo una capa, la red tiene múltiples capas apiladas.

Así que un MLP es básicamente una red hecha de muchas unidades tipo perceptrón dispuestas en varias capas.

## Cómo funciona

## Idea de alto nivel

Un MLP aprende un mapeo de entradas a salidas.

Ejemplo:

- entrada: edad, ingresos, puntaje crediticio.
- salida: aprobar préstamo o rechazar préstamo.

El modelo comienza con pesos aleatorios. Durante el entrenamiento, compara sus predicciones con las respuestas correctas y ajusta lentamente sus pesos para hacer mejores predicciones.

## Modelo mental para principiantes

Piensa en cada capa como un paso de transformación:

1. Las características de entrada entran en la primera capa.
2. Esa capa mezcla las características usando sumas ponderadas.
3. Una función de activación añade no linealidad.
4. El resultado se pasa a la siguiente capa.
5. Después de varias capas, la capa final produce la predicción.

Sin activaciones, apilar capas se comportaría casi como una gran transformación lineal. Las funciones de activación son las que permiten a la red aprender patrones más complejos.

## Vista de bajo nivel

Para una neurona, la operación es:

```text
salida = activacion(w1*x1 + w2*x2 + ... + wn*xn + sesgo)
```

Donde:

- `x1, x2, ..., xn` son valores de entrada.
- `w1, w2, ..., wn` son pesos aprendidos.
- `sesgo` (bias) es un valor extra aprendido.
- `activacion(...)` es usualmente algo como ReLU, Sigmoid o Tanh.

Para una capa completa, PyTorch típicamente usa `nn.Linear`, que aplica esta transformación ponderada a todas las neuronas de esa capa.

## Cómo funciona el entrenamiento

Entrenar un MLP suele seguir este bucle:

1. Pasar un batch de entradas a través de la red.
2. Obtener predicciones.
3. Comparar las predicciones con las etiquetas reales usando una función de pérdida.
4. Calcular gradientes con retropropagación.
5. Actualizar los pesos usando un optimizador.
6. Repetir muchas veces.

En PyTorch, los pasos principales son:

- `model(x)` para la pasada hacia adelante (forward pass).
- `loss = loss_fn(preds, targets)`
- `loss.backward()`
- `optimizer.step()`
- `optimizer.zero_grad()`

## Arquitectura básica

Un MLP pequeño a menudo se ve así:

```text
Entrada -> Lineal -> ReLU -> Lineal -> ReLU -> Lineal -> Salida
```

## Formas y Dimensiones (Shapes & Dimensions)

Entender las formas de los tensores es el desafío más común al construir MLPs en PyTorch.

### 1. La forma de entrada
Un MLP espera un tensor 2D de forma `(batch_size, num_features)`.
- **`batch_size`**: Número de ejemplos procesados a la vez (ej. 32).
- **`num_features`**: Número de variables de entrada por ejemplo (ej. 4 para Iris).

Si tienes un solo ejemplo, aún debe ser 2D: `(1, num_features)`.

### 2. Dimensiones de la capa lineal
Una capa `nn.Linear(in_features, out_features)` requiere:
- La última dimensión del tensor de entrada debe coincidir con `in_features`.
- La capa producirá un tensor de salida de forma `(batch_size, out_features)`.

### 3. Apilando capas
Al apilar capas, las `out_features` de una capa DEBEN coincidir con las `in_features` de la siguiente:
- `capa1 = nn.Linear(4, 16)`
- `capa2 = nn.Linear(16, 8)`  <-- ¡El 16 coincide!

### 4. La forma de salida
- **Clasificación binaria**: Típicamente `(batch_size, 1)` con `Sigmoid`.
- **Clasificación multiclase**: `(batch_size, num_classes)` con `CrossEntropyLoss`.
- **Regresión**: `(batch_size, 1)` (o más si se predicen múltiples valores).

## Casos de uso de MLP

Los MLPs son más útiles cuando tus datos ya están en un vector de características de tamaño fijo.

Casos de uso comunes:

- **Clasificación y regresión tabular**
  - detección de fraude
  - predicción de pérdida de clientes (churn)
  - predicción de precios de casas

- **Fusión de "embeddings" multimodales**
  - Combinar características de salida de diferentes modelos (ej. embeddings de texto e imagen) en una sola clasificación.

- **Aprendizaje de interacción de características no lineales**
  - Cuando tienes un conjunto pequeño de características (ej. < 100) pero sospechas que la relación entre ellas y la salida es altamente compleja y no lineal.

- **Aprendizaje en línea y a gran escala**
  - Los MLPs pueden entrenarse incrementalmente (mini-batch) en conjuntos de datos masivos que no caben en memoria, mientras que algunos modelos clásicos como SVM o ciertas implementaciones de árboles pueden ser intensivos en memoria.

- **Destilación de conocimiento / Modelos "estudiante"**
  - Modelos grandes y complejos (como Transformers profundos o ensambles) a menudo se "comprimen" en un pequeño MLP estudiante que es mucho más rápido de ejecutar en producción manteniendo la mayor parte del rendimiento.

Los MLPs usualmente **no** son la mejor primera opción para:

- imágenes en bruto (raw pixels)
- secuencias largas de texto
- formas de onda de audio
- series temporales muy largas con dependencias temporales

Para esos casos, las CNNs, RNNs/LSTMs o Transformers suelen ser mejores opciones.

## Cuándo los MLPs suelen superar al ML clásico

Aunque Random Forests y XGBoost son los "reyes de los datos tabulares", los MLPs a menudo pueden tomar la delantera en estos escenarios específicos:

### 1. Datos tabulares de alta dimensión con patrones ocultos
Cuando la relación entre las características no es solo "si la característica X > umbral", sino que implica una compleja "mezcla ponderada" de todas las características simultáneamente.
- **Ejemplo**: Predicción de volatilidad del mercado financiero donde muchas pequeñas señales interactúan de forma continua.

### 2. Extracción de características profundas y aprendizaje de representaciones
Los modelos clásicos trabajan sobre las características que tú les das. Los MLPs **crean nuevas características** en sus capas ocultas. Si las características "puras" no son muy útiles por sí solas pero pueden combinarse en representaciones internas poderosas, un MLP ganará.
- **Ejemplo de dataset**: **MNIST** (cuando se trata como un vector plano de 784 píxeles). Aunque puedes ejecutar un Random Forest sobre píxeles, las capas ocultas de un MLP pueden aprender a agrupar píxeles en "trazos" o "curvas" internamente.

### 3. Entrenamiento en conjuntos de datos masivos (millones de filas)
Los modelos clásicos basados en árboles (como Random Forest) pueden volverse extremadamente lentos de construir a medida que crece el número de filas porque necesitan evaluar muchos puntos de división. Los MLPs prosperan aquí usando **Descenso de Gradiente Estocástico (SGD)**, que solo mira un pequeño batch a la vez.
- **Ejemplo de dataset**: **Higgs Boson Dataset** (más de 7 millones de filas). Los datos de física a gran escala a menudo se benefician de los límites suaves y no lineales que crea un MLP.

### 4. Espacios de entrada/salida continuos
Si tus entradas y salidas son continuas y necesitas una función suave y diferenciable (en lugar de una función de "pasos" como la que proporciona un árbol de decisión), un MLP es la opción superior.
- **Ejemplo de dataset**: **Cinemática inversa** en robótica. Mapear coordenadas $(x, y, z)$ a un conjunto continuo de ángulos de motor.

### 5. Tareas de múltiples salidas (Multi-output)
Si necesitas predecir múltiples valores relacionados a la vez (ej. predecir 10 métricas meteorológicas diferentes simultáneamente), un solo MLP puede aprender una representación interna compartida para todas ellas, mientras que la mayoría de los modelos de ML clásicos requieren entrenar un modelo separado por cada salida.

## Ejemplo de PyTorch para entrenar un MLP

Este ejemplo entrena un pequeño MLP para clasificación binaria usando datos tabulares sintéticos.

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset sintético: 1000 filas, 4 características de entrada
X = torch.randn(1000, 4)
y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0).float().unsqueeze(1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
).to(device)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        preds = model(xb)
        loss = loss_fn(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"epoch {epoch + 1}, loss = {total_loss:.4f}")
```

## Qué está haciendo este ejemplo

- `nn.Linear(4, 16)` mapea 4 características de entrada a una capa oculta de 16 neuronas.
- `nn.ReLU()` añade no linealidad.
- el `nn.Linear(8, 1)` final produce un valor de salida.
- `nn.Sigmoid()` convierte esa salida en una probabilidad entre 0 y 1.
- `BCELoss` mide el error de clasificación binaria.
- `Adam` desplaza los pesos durante el entrenamiento.

## Buenos conjuntos de datos para entrenar MLPs

A continuación se muestran conjuntos de datos útiles para principiantes agrupados por caso de uso.

## 1. Clasificación tabular

- **Iris**
  - Caso de uso: clasificar especies de flores a partir de medidas numéricas.
  - Por qué es bueno: pequeño, limpio, ideal para principiantes.
  - Enlace: [UCI Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris)

- **Breast Cancer Wisconsin**
  - Caso de uso: clasificar tumores como malignos o benignos a partir de características numéricas.
  - Por qué es bueno: conjunto de datos clásico de clasificación binaria.
  - Enlace: [UCI Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

- **Adult Income**
  - Caso de uso: predecir si los ingresos superan un umbral.
  - Por qué es bueno: introduce el preprocesamiento tabular del mundo real.
  - Enlace: [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)

## 2. Regresión tabular

- **California Housing**
  - Caso de uso: predecir valores de viviendas a partir de características numéricas estructuradas.
  - Por qué es bueno: conjunto de datos de regresión estándar.
  - Enlace: [scikit-learn California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

- **Wine Quality**
  - Caso de uso: predecir el puntaje de calidad del vino a partir de características fisicoquímicas.
  - Por qué es bueno: regresión o clasificación estructurada práctica.
  - Enlace: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)

## 3. Predicción binaria a partir de características de estilo empresarial

- **Titanic**
  - Caso de uso: predecir la supervivencia de los pasajeros a partir de características tabulares.
  - Por qué es bueno: conjunto de datos común para principiantes para ingeniería de características y clasificación.
  - Enlace: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic)

- **Telco Customer Churn**
  - Caso de uso: predecir la pérdida de clientes a partir de la información de la cuenta del cliente.
  - Por qué es bueno: útil para la clasificación binaria de estilo empresarial real.
  - Enlace: [IBM Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## ¿Con qué conjunto de datos deberías empezar?

Para aprender:

- empieza con **Iris** si quieres la tarea de clasificación limpia más pequeña.
- usa **Breast Cancer Wisconsin** para clasificación binaria.
- usa **California Housing** para regresión.
- usa **Titanic** o **Adult Income** cuando quieras un preprocesamiento más realista.

## Resumen

Un MLP es la red neuronal más común para principiantes porque enseña claramente el bucle de entrenamiento central:

- capas
- activaciones
- pérdida
- retropropagación
- optimización

Si tus datos están estructurados en características numéricas de tamaño fijo, un MLP suele ser la primera arquitectura de red neuronal que vale la pena probar.

## ¿Cuándo elegirías un MLP sobre el machine learning clásico?

Esta es una pregunta práctica importante, porque para datos tabulares, los modelos clásicos de machine learning suelen ser muy fuertes.

Podrías elegir un MLP sobre modelos como Regresión Logística, SVM, Random Forest o Gradient Boosting cuando:

- quieres aprender los fundamentos de las redes neuronales en un entorno sencillo.
- tu problema puede beneficiarse de aprender interacciones de características no lineales a través de capas ocultas.
- tu conjunto de datos es lo suficientemente grande como para que una red neuronal tenga espacio para aprender patrones más ricos.
- quieres mantenerte dentro de un flujo de trabajo basado en PyTorch que luego pueda crecer hacia modelos más profundos.
- planeas combinar datos tabulares con embeddings, características de imagen, características de texto u otros componentes neuronales.
- quieres una red neuronal de referencia (baseline) antes de probar arquitecturas más especializadas.

Podrías preferir el ML clásico en cambio cuando:

- el conjunto de datos es tabular de tamaño pequeño o mediano.
- la interpretabilidad importa mucho.
- la velocidad de entrenamiento y la simplicidad importan más que la flexibilidad arquitectónica.
- los modelos basados en árboles ya funcionan muy bien.
- necesitas un baseline fuerte con menos esfuerzo de ajuste (tuning).

Una intuición aproximada:

- la **Regresión Logística** suele ser una opción inicial fuerte para problemas de clasificación lineal simples.
- las **SVM** pueden funcionar bien en conjuntos de datos más pequeños con límites de clase claros.
- **Random Forest** y **Gradient Boosting** suelen ser excelentes para datos tabulares empresariales.
- el **MLP** se vuelve más atractivo cuando quieres un aprendizaje de representaciones basado en redes neuronales o un camino hacia sistemas de deep learning más grandes.

Para muchos problemas tabulares reales, el mejor enfoque de ingeniería es:

1. empezar con un baseline simple como Regresión Logística o Random Forest.
2. medir el rendimiento.
3. probar un MLP si tienes razones para creer que el aprendizaje de representaciones ocultas puede ayudar.
4. quedarse con el modelo que mejor funcione bajo una evaluación realista.

Así que la respuesta honesta es:

- elige un **MLP** cuando quieras una solución de red neuronal, tengas suficientes datos o esperes que el aprendizaje de capas ocultas importe.
- elige **ML clásico** cuando quieras baselines rápidos, fuertes e interpretables para datos estructurados.
