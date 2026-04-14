import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Device Selection
# This ensures the script runs on GPU (CUDA) if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Define the Model Architecture
# Inheriting from nn.Module is the standard way to build models in PyTorch.
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        # First Fully Connected Layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation Function
        self.relu = nn.ReLU()
        # Second Fully Connected Layer
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Sigmoid for binary classification probability
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Define the data flow through the layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 3. Generate Synthetic Data
# We create 1000 samples with 4 features each.
# The target is 1 if the sum of features is positive, 0 otherwise.
torch.manual_seed(42)
X = torch.randn(1000, 4)
y = (X.sum(dim=1) > 0).float().unsqueeze(1)

# Wrap in a Dataset and DataLoader for batching
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Initialize Model, Loss, and Optimizer
model = SimpleMLP(input_size=4, hidden_size=16, output_size=1).to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Training Loop
num_epochs = 10
print("Starting training...")

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, targets in dataloader:
        # Move data to the same device as the model
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward Pass: Compute predictions
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward Pass: Compute gradients
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Backpropagation
        optimizer.step()       # Update weights

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training complete.")

# 6. Basic Inference Test
test_input = torch.randn(1, 4).to(device)
with torch.no_grad():
    prediction = model(test_input)
    print(f"Test Input: {test_input.cpu().numpy()}")
    print(f"Model Prediction (Probability): {prediction.item():.4f}")
