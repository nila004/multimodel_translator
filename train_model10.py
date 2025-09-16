# train_model.py
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Data
train_df = pd.read_csv("data/sign_mnist_train.csv")
valid_df = pd.read_csv("data/sign_mnist_test.csv")

IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1

# ✅ Fix labels (remove J=9 and Z=25, shift others)
def fix_labels(y):
    y_fixed = []
    for label in y:
        if label == 9 or label == 25:
            continue
        elif 9 < label < 25:
            y_fixed.append(label - 1)
        elif label > 25:
            y_fixed.append(label - 2)
        else:
            y_fixed.append(label)
    return np.array(y_fixed)

# Custom Dataset
class AslDataset(Dataset):
    def __init__(self, base_df, augment=False):
        x_df = base_df.copy()
        y_df = x_df.pop('label')
        y_df = fix_labels(y_df.values)

        x_df = x_df.values / 255.0
        x_df = x_df.reshape(-1, IMG_CHS, IMG_WIDTH, IMG_HEIGHT)

        self.xs = torch.tensor(x_df).float()
        self.ys = torch.tensor(y_df).long()

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(28, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = None

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.xs)

# Dataloaders
BATCH_SIZE = 32
train_data = AslDataset(train_df, augment=True)
valid_data = AslDataset(valid_df, augment=False)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)

# Classes
n_classes = 24  # A–Y (no J, Z)
flattened_img_size = 75 * 3 * 3

# Model
model = nn.Sequential(
    nn.Conv2d(IMG_CHS, 25, 3, stride=1, padding=1),
    nn.BatchNorm2d(25),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),

    nn.Conv2d(25, 50, 3, stride=1, padding=1),
    nn.BatchNorm2d(50),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.MaxPool2d(2, stride=2),

    nn.Conv2d(50, 75, 3, stride=1, padding=1),
    nn.BatchNorm2d(75),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),

    nn.Flatten(),
    nn.Linear(flattened_img_size, 512),
    nn.Dropout(0.4),
    nn.ReLU(),
    nn.Linear(512, n_classes)
).to(device)

# Loss and Optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Accuracy Function
def get_batch_accuracy(output, y):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct

# Validation
def validate():
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            val_loss += loss_function(output, y).item()
            correct += get_batch_accuracy(output, y)
    val_loss /= len(valid_loader)
    val_acc = correct / len(valid_loader.dataset)
    return val_loss, val_acc

# Training
def train():
    model.train()
    train_loss = 0
    correct = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += get_batch_accuracy(output, y)
    train_loss /= len(train_loader)
    train_acc = correct / len(train_loader.dataset)
    return train_loss, train_acc

# Training Loop
epochs = 30
best_acc = 0
patience, patience_counter = 5, 0

for epoch in range(epochs):
    train_loss, train_acc = train()
    val_loss, val_acc = validate()
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_sign_model.pth")
        print("✅ Best model saved!")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

print(f"Best Validation Accuracy: {best_acc:.4f}")
