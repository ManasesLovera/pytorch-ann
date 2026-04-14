# 🎯 Project Roadmap & Tasks

## 🚀 Current Focus
- [ ] **Complete MLP (Multi-Layer Perceptron) Module**
  - High-priority implementation of foundational scripts and datasets.

---

## 🛠️ To-Do

### Phase 1: MLP (Multi-Layer Perceptron)
- [x] **Task 1: Basic training script** (`learning/mlp/train_basic.py`)
  - [x] Implement `nn.Module` class for the architecture.
  - [x] Include automatic CUDA device selection.
  - [x] Use synthetic data for a zero-setup run.
- [x] **Task 2: Real dataset loader** (`learning/mlp/datasets/iris_loader.py`)
  - [x] Use `scikit-learn` to fetch the Iris dataset.
  - [x] Implement `StandardScaler` for feature normalization.
  - [x] Wrap in a PyTorch `DataLoader`.
- [x] **Task 3: Binary classification use-case** (`learning/mlp/use-cases/binary_classification.py`)
  - [x] Implement a full training and evaluation loop.
  - [x] Add an "Inference" section to test the model on new inputs.
- [ ] **Task 4: Documentation updates**
  - [ ] Add "Shapes & Dimensions" section to `learning/mlp/README.md`.

### Phase 2: CNN (Convolutional Neural Networks)
- [ ] Create `learning/cnn/README.md` with core concepts.
- [ ] Implement basic CNN for image classification (MNIST/CIFAR).

### Phase 3: RNN & LSTM
- [ ] Create folder structures and conceptual READMEs.
- [ ] Implement sequence prediction examples.

---

## ✅ Completed
- [x] Initial project scaffold created.
- [x] CUDA availability verified (`test.py`).
- [x] Basic CUDA tensor operations documented (`cuda_basics.py`).
- [x] Conceptual MLP guide written (`learning/mlp/README.md`).
