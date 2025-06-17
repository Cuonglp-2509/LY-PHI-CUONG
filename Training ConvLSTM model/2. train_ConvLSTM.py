import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Settings
DATA_DIR = "data"
MODEL_DIR = "models"
FIGURE_DIR = "figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 17
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
PATIENCE = 5  # Early stopping patience

# Data augmentation function
def augment_keypoints(keypoints, rotation_range=0.1, scale_range=0.1, noise_std=0.01):
    # keypoints: [T, V, C] e.g., [30, 17, 2]
    T, V, C = keypoints.shape
    # Rotation
    theta = np.random.uniform(-rotation_range, rotation_range)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    keypoints = np.matmul(keypoints.reshape(T*V, C), rotation_matrix).reshape(T, V, C)
    # Scaling
    scale = np.random.uniform(1-scale_range, 1+scale_range)
    keypoints *= scale
    # Noise
    keypoints += np.random.normal(0, noise_std, keypoints.shape)
    return keypoints

# Custom dataset with augmentation
class KeypointDataset(TensorDataset):
    def __init__(self, inputs, labels, augment=True):
        super().__init__(inputs, labels)
        self.augment = augment

    def __getitem__(self, idx):
        x, y = self.tensors[0][idx], self.tensors[1][idx]
        if self.augment:
            x_np = x.numpy().reshape(SEQUENCE_LENGTH, NUM_KEYPOINTS, 2)
            x_np = augment_keypoints(x_np)
            x = torch.tensor(x_np.reshape(SEQUENCE_LENGTH, NUM_KEYPOINTS*2), dtype=torch.float32)
        return x, y
    

class ConvLSTM_Model(nn.Module):
    def __init__(self, in_channels=2, num_class=12, sequence_length=30, num_keypoints=17):
        super(ConvLSTM_Model, self).__init__()
        self.sequence_length = sequence_length
        self.num_keypoints = num_keypoints

        # 1D Convolutional layers with residual connections
        self.conv1 = nn.Conv1d(in_channels * num_keypoints, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout(0.3)

        # Residual adapters
        self.res_adapter1 = nn.Conv1d(in_channels * num_keypoints, 64, kernel_size=1) if in_channels * num_keypoints != 64 else nn.Identity()
        self.res_adapter2 = nn.Conv1d(64, 128, kernel_size=1) if 64 != 128 else nn.Identity()

        # LSTM input size
        lstm_in_size = 128  # Output channels from conv4

        # LSTM layer
        self.lstm = nn.LSTM(lstm_in_size, 256, num_layers=2, batch_first=True, dropout=0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_class)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        # Input x: [B, T, V*C] -> Reshape to [B, V*C, T]
        B, T, VC = x.shape
        x = x.view(B, T, self.num_keypoints, -1)  # [B, T, V, C]
        x = x.permute(0, 3, 2, 1).reshape(B, VC, T)  # [B, V*C, T]

        # Convolutional layers with residual connections
        residual = self.res_adapter1(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + residual  # Residual connection
        x = self.pool(x)  # [B, 64, T//2]
        x = self.dropout_conv(x)

        residual = self.res_adapter2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = x + residual  # Residual connection
        x = self.pool(x)  # [B, 128, T//4]
        x = self.dropout_conv(x)

        # Reshape for LSTM
        x = x.permute(0, 2, 1)  # [B, T//4, 128]

        # LSTM layer
        lstm_out, _ = self.lstm(x)  # [B, T//4, 256]
        x = lstm_out[:, -1, :]  # Last time step [B, 256]

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x

# Load dataset
def load_dataset():
    try:
        X = np.load(os.path.join(DATA_DIR, "processed_features.npy"))  # [N, 30, 17, 2]
        Y = np.load(os.path.join(DATA_DIR, "processed_features_labels.npy"))  # [N]
        with open(os.path.join(DATA_DIR, "label_mapping.pkl"), "rb") as f:
            label_map = pickle.load(f)
        labels = sorted(set(label_map.values()))
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        Y_encoded = np.array([label_to_idx.get(label, -1) for label in Y])
        if -1 in Y_encoded:
            raise ValueError("Some labels in Y are not found in label_map")
        return X, Y_encoded, labels
    except FileNotFoundError as e:
        print(f"Error: Data file not found - {e}")
        exit(1)
    except ValueError as ve:
        print(f"Error: Invalid label encountered - {ve}")
        exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

# Compute class weights
def compute_class_weights(Y):
    counts = np.bincount(Y)
    weights = 1. / (counts + 1e-6)  # Avoid division by zero
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, label_names, epoch):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"Confusion Matrix - Epoch {epoch+1}")
    plt.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURE_DIR, f"confusion_matrix_epoch_{epoch+1}.png"))
    plt.close()

# Main training loop
if __name__ == "__main__":
    # Load dataset
    X, Y, label_names = load_dataset()
    print(f"Loaded dataset: X shape={X.shape}, Y shape={Y.shape}, Classes={len(label_names)}")

    # Reshape X for ConvLSTM: [N, 30, 17*2]
    X = X.reshape(X.shape[0], SEQUENCE_LENGTH, NUM_KEYPOINTS * 2)

    # Split into train and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=42
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.long)

    # Create data loaders
    train_dataset = KeypointDataset(X_train, Y_train, augment=True)
    val_dataset = KeypointDataset(X_val, Y_val, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = ConvLSTM_Model(
        in_channels=2,
        num_class=len(label_names),
        sequence_length=SEQUENCE_LENGTH,
        num_keypoints=NUM_KEYPOINTS
    ).to(DEVICE)

    # Compute class weights
    class_weights = compute_class_weights(Y_train.numpy())
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "convlstm_model.pth")

    # Early stopping variables
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        loss_sum = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        avg_loss = loss_sum / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item()
                _, predicted = torch.max(out, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # Plot confusion matrix
        plot_confusion_matrix(all_labels, all_preds, label_names, epoch)

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
            torch.save(best_model_state, model_path)
            print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best model saved to {model_path}")
    print(f"Confusion matrices saved to {FIGURE_DIR}")