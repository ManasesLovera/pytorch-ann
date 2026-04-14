# MLP: Multi-Layer Perceptron

## Good to know first

Before learning MLPs, it helps to understand a few core ideas that show up across almost all neural networks.

## What is a neural network?

A neural network is a model that learns patterns by adjusting many numeric parameters called **weights**.

Instead of writing explicit rules by hand, you give the model examples:

- inputs
- expected outputs

During training, the model changes its internal weights so its predictions get closer to the correct answers.

At a very high level, a neural network is just a stack of mathematical functions that transforms input data into an output.

Example:

- input: a person's age, salary, and credit score
- output: probability of loan approval

The network learns how those inputs relate to the output by seeing many examples.

## How is a neural network different from classic machine learning models?

Classic machine learning and neural networks both learn from data, but they usually differ in how much feature learning they do on their own.

Traditional models often include:

- linear regression
- logistic regression
- decision trees
- random forests
- gradient boosting

These models often work very well on structured tabular data, especially when the features are already useful.

Neural networks differ because they can learn **intermediate representations** inside hidden layers instead of relying only on manually designed features.

In simple terms:

- classic ML often depends more on feature engineering
- neural networks often learn useful internal features automatically

That said, neural networks are not automatically better. For tabular data, classic ML models are often competitive or even stronger than an MLP, especially on smaller datasets.

## What is a feature?

A **feature** is one input variable used by the model.

Examples:

- age
- salary
- number of previous purchases
- pixel intensity
- word embedding value

If your dataset has 10 columns used as inputs, then your model has 10 features.

## What is a neuron?

A neuron is a small computation unit inside the network.

It:

1. receives input values
2. multiplies them by weights
3. adds them together with a bias
4. applies an activation function
5. passes the result forward

This is inspired loosely by biology, but in machine learning it is just math.

## What is a layer?

A layer is a group of neurons operating at the same stage.

The common layer types in an MLP are:

- **input layer**: receives the features
- **hidden layers**: transform the information
- **output layer**: produces the final prediction

When people say a model is "deep," they usually mean it has many hidden layers.

## What does feedforward mean?

**Feedforward** means data moves in one direction only:

```text
input -> hidden layers -> output
```

There are no loops through time and no internal memory of previous steps.

This is different from recurrent models like RNNs and LSTMs, where earlier sequence steps can affect later ones through a hidden state.

MLPs are feedforward networks.

## What is an activation function?

An activation function is the non-linear part of a neuron.

Common examples:

- `ReLU`
- `Sigmoid`
- `Tanh`

Without activation functions, multiple linear layers would collapse into something similar to a single linear transformation. That would make the network much less expressive.

The activation function is what helps a neural network learn more complex patterns.

## What is a prediction?

A prediction is simply the model's output for a given input.

Examples:

- a probability of class 1
- one of several class scores
- a house price
- a future numeric value

## What is a loss function?

A loss function measures how wrong the model's predictions are.

Examples:

- `BCELoss` for binary classification
- `CrossEntropyLoss` for multi-class classification
- `MSELoss` for regression

Training tries to reduce this loss over time.

## What is backpropagation?

Backpropagation is the method used to compute how each weight contributed to the error.

After the model makes a prediction and the loss is computed:

1. PyTorch calculates gradients
2. each gradient tells you how a weight should change to reduce the loss
3. the optimizer uses those gradients to update the weights

You do not usually write backpropagation by hand in PyTorch. Calling:

```python
loss.backward()
```

asks PyTorch to compute those gradients automatically.

## What is a gradient?

A gradient tells you how sensitive the loss is to a parameter.

If a small change in a weight causes the loss to change a lot, that weight has a large gradient.

Gradients are the signals used to improve the model during training.

## What is an optimizer?

An optimizer updates the model weights after gradients are computed.

Common choices:

- `SGD`
- `Adam`

The optimizer is the part that actually changes the weights from one training step to the next.

## What is an epoch and a batch?

These terms appear constantly in deep learning:

- **batch**: a small group of training examples processed together
- **epoch**: one full pass through the entire training dataset

Example:

- dataset size: 1000 rows
- batch size: 100
- batches per epoch: 10

Training for 20 epochs means the model sees the whole dataset 20 times.

## Why use batches instead of the whole dataset at once?

Batches are used because:

- they fit better in memory
- they make GPU training practical
- they often help optimization behave better

## Why does CUDA matter here?

PyTorch can train neural networks on either:

- CPU
- GPU

For many neural network workloads, a GPU is much faster because it can process many numeric operations in parallel.

That is why moving tensors and models to `cuda` is so common in PyTorch code.

## What is MLP?

An MLP, or Multi-Layer Perceptron, is one of the most basic neural network architectures.

It is a **feedforward neural network**, which means information moves from the input layer, through one or more hidden layers, to the output layer. It does not loop backward through time like an RNN, and it does not scan local image regions like a CNN.

An MLP is usually built from **fully connected layers**. That means every neuron in one layer connects to every neuron in the next layer.

In practice, MLPs are commonly used for:

- tabular data
- basic classification
- regression
- small structured datasets

## Why is it called Multi-Layer Perceptron?

The name has three parts:

- **Perceptron**: the original simple neuron-like model that takes inputs, applies weights, adds a bias, and produces an output
- **Layer**: neurons are grouped into layers
- **Multi-Layer**: instead of only one layer, the network has multiple layers stacked together

So an MLP is basically a network made of many perceptron-like units arranged in several layers.

## How it works

## High-level idea

An MLP learns a mapping from inputs to outputs.

Example:

- input: age, income, credit score
- output: approve loan or reject loan

The model starts with random weights. During training, it compares its predictions against the correct answers and slowly adjusts its weights to make better predictions.

## Beginner mental model

Think of each layer as a transformation step:

1. The input features go into the first layer.
2. That layer mixes the features using weighted sums.
3. An activation function adds non-linearity.
4. The result is passed to the next layer.
5. After several layers, the final layer produces the prediction.

Without activations, stacking layers would behave almost like one big linear transformation. The activation functions are what allow the network to learn more complex patterns.

## Lower-level view

For one neuron, the operation is:

```text
output = activation(w1*x1 + w2*x2 + ... + wn*xn + bias)
```

Where:

- `x1, x2, ..., xn` are input values
- `w1, w2, ..., wn` are learned weights
- `bias` is an extra learned value
- `activation(...)` is usually something like ReLU, Sigmoid, or Tanh

For a full layer, PyTorch typically uses `nn.Linear`, which applies this weighted transformation to all neurons in that layer.

## How training works

Training an MLP usually follows this loop:

1. Pass a batch of inputs through the network.
2. Get predictions.
3. Compare predictions with the true labels using a loss function.
4. Compute gradients with backpropagation.
5. Update the weights using an optimizer.
6. Repeat many times.

In PyTorch, the main steps are:

- `model(x)` for the forward pass
- `loss = loss_fn(preds, targets)`
- `loss.backward()`
- `optimizer.step()`
- `optimizer.zero_grad()`

## Basic architecture

A small MLP often looks like this:

```text
Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
```

## Shapes & Dimensions

Understanding tensor shapes is the most common challenge when building MLPs in PyTorch.

### 1. The Input Shape
An MLP expects a 2D tensor of shape `(batch_size, num_features)`.
- **`batch_size`**: Number of examples processed at once (e.g., 32).
- **`num_features`**: Number of input variables per example (e.g., 4 for Iris).

If you have a single example, it must still be 2D: `(1, num_features)`.

### 2. Linear Layer Dimensions
A `nn.Linear(in_features, out_features)` layer requires:
- The input tensor's last dimension must match `in_features`.
- The layer will produce an output tensor of shape `(batch_size, out_features)`.

### 3. Stacking Layers
When stacking layers, the `out_features` of one layer MUST match the `in_features` of the next:
- `layer1 = nn.Linear(4, 16)`
- `layer2 = nn.Linear(16, 8)`  <-- 16 matches!

### 4. The Output Shape
- **Binary Classification**: Typically `(batch_size, 1)` with `Sigmoid`.
- **Multi-class Classification**: `(batch_size, num_classes)` with `CrossEntropyLoss`.
- **Regression**: `(batch_size, 1)` (or more if predicting multiple values).

## MLP use cases

MLPs are most useful when your data is already in a fixed-size feature vector.

Common use cases:

- **Tabular classification and regression**
  - fraud detection
  - customer churn prediction
  - house price prediction

- **Multi-modal embeddings fusion**
  - Combining output features from different models (e.g., text and image embeddings) into a single classification.

- **Non-linear Feature Interaction Learning**
  - When you have a small set of features (e.g., < 100) but suspect the relationship between them and the output is highly complex and non-linear.

- **Online Learning and Large Scale**
  - MLPs can be trained incrementally (mini-batch) on massive datasets that don't fit in memory, whereas some classic models like SVM or certain Tree implementations can be memory-intensive.

- **Knowledge Distillation / Student Models**
  - Large complex models (like deep Transformers or ensembles) are often "compressed" into a small MLP student that is much faster to run in production while retaining most of the performance.

MLPs are usually **not** the best first choice for:

- raw images
- long text sequences
- audio waveforms
- very long time series with temporal dependencies

For those, CNNs, RNNs/LSTMs, or Transformers are usually better fits.

## When MLPs often beat classic ML

While Random Forests and XGBoost are "kings of tabular data," MLPs can often take the lead in these specific scenarios:

### 1. High-Dimensional Tabular Data with Hidden Patterns
When the relationship between features isn't just "if feature X > threshold," but involves a complex "weighted blend" of all features simultaneously.
- **Example**: Financial market volatility prediction where many small signals interact in a continuous way.

### 2. Deep Feature Extraction and Representation Learning
Classic models work on the features you give them. MLPs **create new features** in their hidden layers. If the "raw" features are not very useful alone but can be combined into powerful internal representations, an MLP will win.
- **Dataset Example**: **MNIST** (when treated as a flat 784-pixel vector). While you *can* run a Random Forest on pixels, an MLP's hidden layers can learn to group pixels into "strokes" or "curves" internally.

### 3. Training on Massive Datasets (Millions of rows)
Classic tree-based models (like Random Forest) can become extremely slow to build as the number of rows grows because they need to evaluate many split points. MLPs thrive here using **Stochastic Gradient Descent (SGD)**, which only looks at a small batch at a time.
- **Dataset Example**: **Higgs Boson Dataset** (7 million+ rows). Large-scale physics data often benefits from the smooth, non-linear boundaries an MLP creates.

### 4. Continuous Input/Output Spaces
If your inputs and outputs are continuous and you need a smooth, differentiable function (rather than a "step" function like a Decision Tree provides), an MLP is the superior choice.
- **Dataset Example**: **Inverse Kinematics** in Robotics. Mapping $(x, y, z)$ coordinates to a continuous set of motor angles.

### 5. Multi-Output Tasks
If you need to predict multiple related values at once (e.g., predicting 10 different weather metrics simultaneously), a single MLP can learn a shared internal representation for all of them, whereas most classic ML models require training one separate model per output.

For those, CNNs, RNNs/LSTMs, or Transformers are usually better fits.

## PyTorch example for training an MLP

This example trains a small MLP for binary classification using synthetic tabular data.

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

## What this example is doing

- `nn.Linear(4, 16)` maps 4 input features into a hidden layer of 16 neurons
- `nn.ReLU()` adds non-linearity
- the final `nn.Linear(8, 1)` produces one output value
- `nn.Sigmoid()` converts that output into a probability between 0 and 1
- `BCELoss` measures binary classification error
- `Adam` updates the weights during training

## Good datasets for training MLPs

Below are useful beginner datasets grouped by use case.

## 1. Tabular classification

- **Iris**
  - Use case: classify flower species from numeric measurements
  - Why good: tiny, clean, beginner-friendly
  - Link: [UCI Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris)

- **Breast Cancer Wisconsin**
  - Use case: classify tumors as malignant or benign from numeric features
  - Why good: classic binary classification dataset
  - Link: [UCI Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

- **Adult Income**
  - Use case: predict whether income exceeds a threshold
  - Why good: introduces real-world tabular preprocessing
  - Link: [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)

## 2. Tabular regression

- **California Housing**
  - Use case: predict housing values from structured numeric features
  - Why good: standard regression dataset
  - Link: [scikit-learn California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

- **Wine Quality**
  - Use case: predict wine quality score from physicochemical features
  - Why good: practical structured regression or classification
  - Link: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)

## 3. Binary prediction from business-style features

- **Titanic**
  - Use case: predict passenger survival from tabular features
  - Why good: common beginner dataset for feature engineering and classification
  - Link: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic)

- **Telco Customer Churn**
  - Use case: predict churn from customer account information
  - Why good: useful for real business-style binary classification
  - Link: [IBM Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Which dataset should you start with?

For learning:

- start with **Iris** if you want the smallest clean classification task
- use **Breast Cancer Wisconsin** for binary classification
- use **California Housing** for regression
- use **Titanic** or **Adult Income** when you want more realistic preprocessing

## Summary

An MLP is the most common beginner neural network because it teaches the core training loop clearly:

- layers
- activations
- loss
- backpropagation
- optimization

If your data is structured into fixed-size numeric features, an MLP is usually the first neural network architecture worth trying.

## When would you choose an MLP over classic machine learning?

This is an important practical question, because for tabular data, classic machine learning models are often very strong.

You might choose an MLP over models like Logistic Regression, SVM, Random Forest, or Gradient Boosting when:

- you want to learn neural network fundamentals in a simple setting
- your problem may benefit from learning non-linear feature interactions through hidden layers
- your dataset is large enough that a neural network has room to learn richer patterns
- you want to stay inside a PyTorch-based workflow that may later grow into deeper models
- you plan to combine tabular data with embeddings, image features, text features, or other neural components
- you want a baseline neural network before trying more specialized architectures

You might prefer classic ML instead when:

- the dataset is small or medium-sized tabular data
- interpretability matters a lot
- training speed and simplicity matter more than architectural flexibility
- tree-based models already perform very well
- you need a strong baseline with less tuning effort

A rough intuition:

- **Logistic Regression** is often a strong first choice for simple linear classification problems
- **SVM** can work well on smaller datasets with clear class boundaries
- **Random Forest** and **Gradient Boosting** are often excellent for tabular business data
- **MLP** becomes more attractive when you want neural-network-based representation learning or a path toward larger deep learning systems

For many real tabular problems, the best engineering approach is:

1. start with a simple baseline such as Logistic Regression or Random Forest
2. measure performance
3. try an MLP if you have reason to believe learned hidden representations may help
4. keep the model that performs best under realistic evaluation

So the honest answer is:

- choose an **MLP** when you want a neural-network solution, have enough data, or expect hidden-layer learning to matter
- choose **classic ML** when you want fast, strong, interpretable baselines for structured data
