# MLP: Perceptron Multicapa

## Bueno saber primero

Antes de aprender MLPs, ayuda entender algunas ideas clave que aparecen en casi todas las redes neuronales.

## ¿Qué es una red neuronal?

Una red neuronal es un modelo que aprende patrones ajustando muchos parámetros numéricos llamados **pesos**.

En lugar de escribir reglas explícitas a mano, le das al modelo ejemplos:

- entradas
- salidas esperadas

Durante el entrenamiento, el modelo cambia sus pesos internos para que sus predicciones se acerquen más a las respuestas correctas.

A un nivel muy alto, una red neuronal es solo una pila de funciones matemáticas que transforma datos de entrada en una salida.

Ejemplo:

- entrada: edad, salario y puntaje de crédito de una persona
- salida: probabilidad de aprobar un préstamo

La red aprende cómo esas entradas se relacionan con la salida viendo muchos ejemplos.

## ¿En qué se diferencia una red neuronal de los modelos clásicos de machine learning?

El machine learning clásico y las redes neuronales aprenden de los datos, pero normalmente se diferencian en cuánto aprendizaje de características hacen por sí solos.

Los modelos tradicionales suelen incluir:

- regresión lineal
- regresión logística
- árboles de decisión
- random forests
- gradient boosting

Estos modelos suelen funcionar muy bien en datos tabulares estructurados, especialmente cuando las variables ya son útiles.

Las redes neuronales se diferencian porque pueden aprender **representaciones intermedias** dentro de las capas ocultas en lugar de depender solo de características diseñadas manualmente.

En términos simples:

- el ML clásico suele depender más de la ingeniería de características
- las redes neuronales suelen aprender características internas útiles automáticamente

Dicho eso, las redes neuronales no son automáticamente mejores. Para datos tabulares, los modelos clásicos de ML suelen ser competitivos o incluso más fuertes que un MLP, especialmente en datasets pequeños.

## ¿Qué es una característica o feature?

Una **feature** es una variable de entrada usada por el modelo.

Ejemplos:

- edad
- salario
- número de compras previas
- intensidad de un píxel
- valor de un embedding de palabras

Si tu dataset tiene 10 columnas usadas como entrada, entonces tu modelo tiene 10 features.

## ¿Qué es una neurona?

Una neurona es una pequeña unidad de cómputo dentro de la red.

Hace lo siguiente:

1. recibe valores de entrada
2. los multiplica por pesos
3. los suma junto con un sesgo o bias
4. aplica una función de activación
5. pasa el resultado hacia adelante

Esto está inspirado de forma muy general en la biología, pero en machine learning es simplemente matemáticas.

## ¿Qué es una capa?

Una capa es un grupo de neuronas que operan en la misma etapa.

Los tipos de capa más comunes en un MLP son:

- **capa de entrada**: recibe las features
- **capas ocultas**: transforman la información
- **capa de salida**: produce la predicción final

Cuando la gente dice que un modelo es "deep", normalmente quiere decir que tiene muchas capas ocultas.

## ¿Qué significa feedforward?

**Feedforward** significa que los datos se mueven en una sola dirección:

```text
entrada -> capas ocultas -> salida
```

No hay bucles a través del tiempo ni memoria interna de pasos anteriores.

Esto es diferente de modelos recurrentes como RNNs y LSTMs, donde pasos anteriores de la secuencia pueden afectar pasos posteriores mediante un estado oculto.

Los MLPs son redes feedforward.

## ¿Qué es una función de activación?

Una función de activación es la parte no lineal de una neurona.

Ejemplos comunes:

- `ReLU`
- `Sigmoid`
- `Tanh`

Sin funciones de activación, varias capas lineales se colapsarían en algo parecido a una sola transformación lineal. Eso haría la red mucho menos expresiva.

La función de activación es lo que ayuda a una red neuronal a aprender patrones más complejos.

## ¿Qué es una predicción?

Una predicción es simplemente la salida del modelo para una entrada dada.

Ejemplos:

- una probabilidad de la clase 1
- uno de varios puntajes de clase
- el precio de una casa
- un valor numérico futuro

## ¿Qué es una función de pérdida?

Una función de pérdida mide qué tan equivocadas están las predicciones del modelo.

Ejemplos:

- `BCELoss` para clasificación binaria
- `CrossEntropyLoss` para clasificación multiclase
- `MSELoss` para regresión

El entrenamiento intenta reducir esta pérdida con el tiempo.

## ¿Qué es backpropagation?

Backpropagation es el método usado para calcular cómo cada peso contribuyó al error.

Después de que el modelo hace una predicción y se calcula la pérdida:

1. PyTorch calcula los gradientes
2. cada gradiente indica cómo debería cambiar un peso para reducir la pérdida
3. el optimizador usa esos gradientes para actualizar los pesos

Normalmente no escribes backpropagation a mano en PyTorch. Llamar a:

```python
loss.backward()
```

le pide a PyTorch que calcule esos gradientes automáticamente.

## ¿Qué es un gradiente?

Un gradiente indica qué tan sensible es la pérdida con respecto a un parámetro.

Si un pequeño cambio en un peso hace que la pérdida cambie mucho, entonces ese peso tiene un gradiente grande.

Los gradientes son las señales usadas para mejorar el modelo durante el entrenamiento.

## ¿Qué es un optimizador?

Un optimizador actualiza los pesos del modelo después de que se calculan los gradientes.

Opciones comunes:

- `SGD`
- `Adam`

El optimizador es la parte que realmente cambia los pesos de un paso de entrenamiento al siguiente.

## ¿Qué es una epoch y un batch?

Estos términos aparecen constantemente en deep learning:

- **batch**: un grupo pequeño de ejemplos de entrenamiento procesados juntos
- **epoch**: una pasada completa por todo el dataset de entrenamiento

Ejemplo:

- tamaño del dataset: 1000 filas
- batch size: 100
- batches por epoch: 10

Entrenar por 20 epochs significa que el modelo ve el dataset completo 20 veces.

## ¿Por qué usar batches en lugar de todo el dataset a la vez?

Los batches se usan porque:

- se ajustan mejor a memoria
- hacen práctico el entrenamiento en GPU
- muchas veces ayudan a que la optimización se comporte mejor

## ¿Por qué importa CUDA aquí?

PyTorch puede entrenar redes neuronales en:

- CPU
- GPU

Para muchas cargas de trabajo de redes neuronales, una GPU es mucho más rápida porque puede procesar muchas operaciones numéricas en paralelo.

Por eso mover tensores y modelos a `cuda` es tan común en código PyTorch.

## ¿Qué es un MLP?

Un MLP, o Multi-Layer Perceptron, es una de las arquitecturas de redes neuronales más básicas.

Es una **red neuronal feedforward**, lo que significa que la información se mueve desde la capa de entrada, pasando por una o más capas ocultas, hasta la capa de salida. No recorre el tiempo hacia atrás como una RNN, ni escanea regiones locales de una imagen como una CNN.

Un MLP normalmente está construido con **capas totalmente conectadas**. Eso significa que cada neurona de una capa se conecta con cada neurona de la siguiente.

En la práctica, los MLPs se usan comúnmente para:

- datos tabulares
- clasificación básica
- regresión
- datasets estructurados pequeños

## ¿Por qué se llama Multi-Layer Perceptron?

El nombre tiene tres partes:

- **Perceptron**: el modelo original simple parecido a una neurona que toma entradas, aplica pesos, suma un bias y produce una salida
- **Layer**: las neuronas se agrupan en capas
- **Multi-Layer**: en vez de tener solo una capa, la red tiene varias capas apiladas

Entonces, un MLP es básicamente una red formada por muchas unidades tipo perceptrón organizadas en varias capas.

## Cómo funciona

## Idea de alto nivel

Un MLP aprende un mapeo de entradas a salidas.

Ejemplo:

- entrada: edad, ingresos y puntaje de crédito
- salida: aprobar o rechazar un préstamo

El modelo empieza con pesos aleatorios. Durante el entrenamiento, compara sus predicciones con las respuestas correctas y va ajustando lentamente sus pesos para mejorar.

## Modelo mental para principiantes

Piensa en cada capa como un paso de transformación:

1. Las features de entrada pasan a la primera capa.
2. Esa capa mezcla las features usando sumas ponderadas.
3. Una función de activación agrega no linealidad.
4. El resultado pasa a la siguiente capa.
5. Después de varias capas, la capa final produce la predicción.

Sin activaciones, apilar capas se comportaría casi como una sola gran transformación lineal. Las funciones de activación son lo que permite a la red aprender patrones más complejos.

## Vista de más bajo nivel

Para una sola neurona, la operación es:

```text
salida = activacion(w1*x1 + w2*x2 + ... + wn*xn + bias)
```

Donde:

- `x1, x2, ..., xn` son valores de entrada
- `w1, w2, ..., wn` son pesos aprendidos
- `bias` es un valor adicional aprendido
- `activacion(...)` suele ser algo como ReLU, Sigmoid o Tanh

Para una capa completa, PyTorch normalmente usa `nn.Linear`, que aplica esta transformación ponderada a todas las neuronas de esa capa.

## Cómo funciona el entrenamiento

Entrenar un MLP normalmente sigue este ciclo:

1. Pasar un batch de entradas por la red.
2. Obtener predicciones.
3. Comparar las predicciones con las etiquetas verdaderas usando una función de pérdida.
4. Calcular gradientes con backpropagation.
5. Actualizar los pesos usando un optimizador.
6. Repetir muchas veces.

En PyTorch, los pasos principales son:

- `model(x)` para el forward pass
- `loss = loss_fn(preds, targets)`
- `loss.backward()`
- `optimizer.step()`
- `optimizer.zero_grad()`

## Arquitectura básica

Un MLP pequeño suele verse así:

```text
Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
```

Ejemplo:

- features de entrada: 10
- capa oculta 1: 64 neuronas
- capa oculta 2: 32 neuronas
- salida: 1 valor para regresión o varios valores para clasificación

## Casos de uso de MLP

Los MLPs son más útiles cuando tus datos ya están en un vector de features de tamaño fijo.

Casos de uso comunes:

- **Clasificación tabular**
  - detección de fraude
  - predicción de churn de clientes
  - aprobación de crédito
  - predicción de riesgo médico

- **Regresión tabular**
  - predicción de precios de casas
  - pronóstico de demanda a partir de features estructuradas
  - estimación de costos de seguros

- **Sistemas simples de recomendación o ranking**
  - predecir probabilidad de clic a partir de features diseñadas

- **Modelo baseline**
  - un buen primer modelo antes de pasar a arquitecturas más complejas

Los MLPs normalmente **no** son la mejor primera opción para:

- imágenes crudas
- secuencias largas de texto
- audio en forma de waveform
- series de tiempo muy largas con dependencias temporales

Para esos casos, CNNs, RNNs/LSTMs o Transformers suelen encajar mejor.

## Ejemplo en PyTorch para entrenar un MLP

Este ejemplo entrena un MLP pequeño para clasificación binaria usando datos tabulares sintéticos.

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Synthetic dataset: 1000 rows, 4 input features
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

- `nn.Linear(4, 16)` mapea 4 features de entrada a una capa oculta de 16 neuronas
- `nn.ReLU()` agrega no linealidad
- la `nn.Linear(8, 1)` final produce un valor de salida
- `nn.Sigmoid()` convierte esa salida en una probabilidad entre 0 y 1
- `BCELoss` mide el error de clasificación binaria
- `Adam` actualiza los pesos durante el entrenamiento

## Buenos datasets para entrenar MLPs

Debajo tienes datasets útiles para principiantes agrupados por caso de uso.

## 1. Clasificación tabular

- **Iris**
  - Caso de uso: clasificar especies de flores a partir de mediciones numéricas
  - Por qué es bueno: pequeño, limpio y amigable para principiantes
  - Link: [UCI Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris)

- **Breast Cancer Wisconsin**
  - Caso de uso: clasificar tumores como malignos o benignos a partir de features numéricas
  - Por qué es bueno: dataset clásico de clasificación binaria
  - Link: [UCI Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

- **Adult Income**
  - Caso de uso: predecir si el ingreso supera un umbral
  - Por qué es bueno: introduce preprocesamiento tabular del mundo real
  - Link: [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)

## 2. Regresión tabular

- **California Housing**
  - Caso de uso: predecir valores de viviendas a partir de features numéricas estructuradas
  - Por qué es bueno: dataset estándar de regresión
  - Link: [scikit-learn California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

- **Wine Quality**
  - Caso de uso: predecir la puntuación de calidad del vino a partir de features fisicoquímicas
  - Por qué es bueno: útil para regresión o clasificación estructurada
  - Link: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)

## 3. Predicción binaria con features de negocio

- **Titanic**
  - Caso de uso: predecir la supervivencia de pasajeros a partir de features tabulares
  - Por qué es bueno: dataset muy común para feature engineering y clasificación
  - Link: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic)

- **Telco Customer Churn**
  - Caso de uso: predecir churn a partir de información de cuentas de clientes
  - Por qué es bueno: útil para clasificación binaria con estilo de negocio real
  - Link: [IBM Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## ¿Con qué dataset deberías empezar?

Para aprender:

- empieza con **Iris** si quieres la tarea de clasificación más pequeña y limpia
- usa **Breast Cancer Wisconsin** para clasificación binaria
- usa **California Housing** para regresión
- usa **Titanic** o **Adult Income** cuando quieras practicar un preprocesamiento más realista

## Resumen

Un MLP es la red neuronal más común para comenzar porque enseña claramente el ciclo central de entrenamiento:

- capas
- activaciones
- pérdida
- backpropagation
- optimización

Si tus datos están estructurados en features numéricas de tamaño fijo, un MLP suele ser la primera arquitectura de red neuronal que vale la pena probar.

## ¿Cuándo elegirías un MLP sobre machine learning clásico?

Esta es una pregunta práctica importante, porque para datos tabulares los modelos clásicos de machine learning suelen ser muy fuertes.

Podrías elegir un MLP sobre modelos como Logistic Regression, SVM, Random Forest o Gradient Boosting cuando:

- quieres aprender fundamentos de redes neuronales en un escenario simple
- tu problema puede beneficiarse de aprender interacciones no lineales entre features mediante capas ocultas
- tu dataset es lo suficientemente grande como para que una red neuronal tenga espacio para aprender patrones más ricos
- quieres mantenerte dentro de un flujo de trabajo basado en PyTorch que luego pueda crecer hacia modelos más profundos
- planeas combinar datos tabulares con embeddings, features de imágenes, features de texto u otros componentes neuronales
- quieres un baseline neuronal antes de probar arquitecturas más especializadas

Podrías preferir ML clásico cuando:

- el dataset tabular es pequeño o mediano
- la interpretabilidad importa mucho
- la velocidad de entrenamiento y la simplicidad importan más que la flexibilidad de la arquitectura
- los modelos basados en árboles ya funcionan muy bien
- necesitas un baseline fuerte con menos esfuerzo de ajuste

Una intuición rápida:

- **Logistic Regression** suele ser una muy buena primera opción para problemas de clasificación lineal simples
- **SVM** puede funcionar muy bien en datasets pequeños con fronteras de clase claras
- **Random Forest** y **Gradient Boosting** suelen ser excelentes para datos tabulares de negocio
- **MLP** se vuelve más atractivo cuando quieres aprendizaje de representaciones con redes neuronales o un camino hacia sistemas de deep learning más grandes

Para muchos problemas tabulares reales, el mejor enfoque de ingeniería es:

1. empezar con un baseline simple como Logistic Regression o Random Forest
2. medir el rendimiento
3. probar un MLP si tienes razones para pensar que aprender representaciones ocultas puede ayudar
4. quedarte con el modelo que mejor rinda bajo una evaluación realista

Entonces, la respuesta honesta es:

- elige un **MLP** cuando quieras una solución basada en redes neuronales, tengas suficientes datos o esperes que el aprendizaje en capas ocultas importe
- elige **ML clásico** cuando quieras baselines rápidos, fuertes e interpretables para datos estructurados
