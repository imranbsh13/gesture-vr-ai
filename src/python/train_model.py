import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# --- Config ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

GESTURES = ["open_hand", "fist", "point_up", "peace", "thumbs_up"]
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- Dataset ---
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Model ---
class GestureClassifier(nn.Module):
    def __init__(self, input_size=63, num_classes=5):
        super(GestureClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# --- Load data ---
def load_data():
    X, y = [], []
    for label, gesture in enumerate(GESTURES):
        path = os.path.join(DATA_DIR, f"{gesture}.npy")
        data = np.load(path)
        X.append(data)
        y.extend([label] * len(data))
        print(f"Loaded {len(data)} samples for '{gesture}' (label {label})")
    return np.vstack(X), np.array(y)

# --- Training loop ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = GestureDataset(X_train, y_train)
    val_dataset = GestureDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = GestureClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    train_losses, val_losses, val_accuracies = [], [], []

    print("\nTraining started...\n")
    for epoch in range(EPOCHS):
        # -- Train --
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # -- Validate --
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()
                correct += (output.argmax(1) == y_batch).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / len(val_dataset) * 100

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Accuracy: {accuracy:.1f}%")

    # -- Save PyTorch model --
    torch_path = os.path.join(MODEL_DIR, "gesture_model.pth")
    torch.save(model.state_dict(), torch_path)
    print(f"\nModel saved to {torch_path}")

    # -- Export to ONNX --
    model.eval()
    dummy_input = torch.randn(1, 63).to(device)
    onnx_path = os.path.join(MODEL_DIR, "gesture_model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=["landmarks"],
        output_names=["gesture"],
        dynamic_axes={"landmarks": {0: "batch_size"}, "gesture": {0: "batch_size"}}
    )
    print(f"ONNX model exported to {onnx_path}")

    # -- Plot training curves --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(val_accuracies, color="green")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy %")

    plot_path = os.path.join(PROCESSED_DIR, "training_curves.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Training curves saved to {plot_path}")

if __name__ == "__main__":
    train()