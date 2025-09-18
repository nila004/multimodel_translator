import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import numpy as np
import torch
import torch.nn as nn
from train_lip_model import LipReadingModel

# ---------------------------
# Load model
# ---------------------------
checkpoint = torch.load("lip_model.pth", map_location="cpu")
vocab = checkpoint["vocab"]
inv_vocab = {v: k for k, v in vocab.items()}

model = LipReadingModel(num_classes=len(vocab))
model.load_state_dict(checkpoint["model"])
model.eval()

# ---------------------------
# Prediction Function
# ---------------------------
def predict_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        frames.append(gray)
    cap.release()

    if len(frames) == 0:
        return "Error: No frames extracted"

    frames = np.array(frames) / 255.0
    frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(1).unsqueeze(0)
    lengths = [frames.shape[1]]

    with torch.no_grad():
        outputs = model(frames, lengths)
        pred = torch.argmax(outputs, dim=1).item()

    return inv_vocab[pred]

# ---------------------------
# GUI
# ---------------------------
def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.avi *.mp4")])
    if file_path:
        word = predict_from_video(file_path)
        messagebox.showinfo("Prediction", f"Predicted word: {word}")

root = tk.Tk()
root.title("Lip Reading Translator")

btn = tk.Button(root, text="Upload Video", command=upload_video, font=("Arial", 14))
btn.pack(pady=20)

root.mainloop()
