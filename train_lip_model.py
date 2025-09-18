import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Dataset Loader
# ---------------------------
class LipReadingDataset(Dataset):
    def __init__(self, root_dir, vocab=None, split="train", val_ratio=0.2):
        self.root_dir = root_dir
        self.folders = sorted(os.listdir(root_dir))

        # Extract word labels (prefix before "_")
        words = [f.split("_")[0] for f in self.folders]

        # Build vocabulary
        if vocab is None:
            self.vocab = {w: i for i, w in enumerate(sorted(set(words)))}
        else:
            self.vocab = vocab

        # Train/val split
        n = len(self.folders)
        split_idx = int(n * (1 - val_ratio))
        if split == "train":
            self.folders = self.folders[:split_idx]
        else:
            self.folders = self.folders[split_idx:]

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        path = os.path.join(self.root_dir, folder)

        # Load frames from video if available
        video_path = None
        for f in os.listdir(path):
            if f.endswith(".avi") or f.endswith(".mp4"):
                video_path = os.path.join(path, f)
                break

        frames = []
        if video_path:
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (64, 64))
                frames.append(gray)
            cap.release()
        else:
            # fallback to frame images
            for file in sorted(os.listdir(path)):
                if file.endswith(".png") or file.endswith(".jpg"):
                    img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (64, 64))
                    frames.append(img)

        frames = np.array(frames) / 255.0
        frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(1)  # [T, 1, 64, 64]

        label_word = folder.split("_")[0]
        label = self.vocab[label_word]

        return frames, label


# ---------------------------
# Collate Function
# ---------------------------
def collate_fn(batch):
    frames, labels = zip(*batch)
    lengths = [len(seq) for seq in frames]
    max_len = max(lengths)

    padded_frames = []
    for seq in frames:
        pad_len = max_len - seq.shape[0]
        padded = torch.cat([seq, torch.zeros(pad_len, 1, 64, 64)], dim=0)
        padded_frames.append(padded)

    padded_frames = torch.stack(padded_frames)
    labels = torch.tensor(labels)

    return padded_frames, labels, lengths


# ---------------------------
# Model: CNN + LSTM
# ---------------------------
class LipReadingModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(64 * 16 * 16, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, lengths):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)
        feats = feats.view(B, T, -1)

        packed = nn.utils.rnn.pack_padded_sequence(feats, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        out = self.fc(h[-1])
        return out


# ---------------------------
# Training Loop
# ---------------------------
def train_model():
    dataset = LipReadingDataset("data/lip_reading", split="train")
    val_dataset = LipReadingDataset("data/lip_reading", vocab=dataset.vocab, split="val")

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = LipReadingModel(num_classes=len(dataset.vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(50):  # adjust epochs
        model.train()
        total_loss = 0
        for frames, labels, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            frames, labels = frames.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(frames, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for frames, labels, lengths in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                outputs = model(frames, lengths)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")

    torch.save({"model": model.state_dict(), "vocab": dataset.vocab}, "lip_model.pth")
    print("✅ Model saved as lip_model.pth")


if __name__ == "__main__":
    train_model()
