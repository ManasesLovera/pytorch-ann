import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_iris_dataloaders(batch_size=16, test_size=0.2, random_state=42):
    """
    Loads the Iris dataset, normalizes features, and returns PyTorch DataLoaders.
    
    Returns:
        train_loader, test_loader, feature_count, class_count
    """
    # 1. Load data from scikit-learn
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 3. Feature Normalization (Critical for MLP performance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 4. Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    # 5. Create DataLoaders
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, X.shape[1], len(iris.target_names)

if __name__ == "__main__":
    # Quick verification run
    train_loader, test_loader, feats, classes = get_iris_dataloaders()
    print(f"Dataset Loaded Successfully!")
    print(f"Features: {feats}")
    print(f"Classes: {classes}")
    
    # Check one batch
    inputs, targets = next(iter(train_loader))
    print(f"Batch shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Sample Normalized Input: {inputs[0]}")
