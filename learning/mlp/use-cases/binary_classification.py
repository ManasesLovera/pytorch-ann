import torch
import torch.nn as nn
import torch.optim as optim
from learning.mlp.datasets.iris_loader import get_iris_dataloaders

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load Data
# Although Iris is multi-class (3), we'll treat it as a classification task
# to demonstrate the full workflow including evaluation.
train_loader, test_loader, input_size, num_classes = get_iris_dataloaders()

# 3. Define the Model
class IrisClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(IrisClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.network(x)

model = IrisClassifier(input_size, 16, num_classes).to(device)

# 4. Loss and Optimizer
# Using CrossEntropyLoss for multi-class (it includes LogSoftmax internally)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Training and Evaluation Loop
def train_and_eval(epochs=50):
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        accuracy = 100 * correct / total
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

train_and_eval()

# 6. Practical Inference Example
print("\n--- Inference Test ---")
# Let's mock a new measurement (normalized)
# [SepalLength, SepalWidth, PetalLength, PetalWidth]
sample_input = torch.tensor([[5.1, 3.5, 1.4, 0.2]], dtype=torch.float32).to(device)
# Note: In a real app, you MUST use the same StandardScaler used in training!

model.eval()
with torch.no_grad():
    prediction = model(sample_input)
    _, predicted_class = torch.max(prediction, 1)
    
    classes = ["Setosa", "Versicolor", "Virginica"]
    print(f"Input: {sample_input.cpu().numpy()}")
    print(f"Predicted Class: {classes[predicted_class.item()]}")
