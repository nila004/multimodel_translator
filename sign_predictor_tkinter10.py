# sign_predictor_tkinter.py
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
import torch.nn as nn

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes (A–Y, skipping J, Z)
classes = [
    'A','B','C','D','E','F','G','H','I',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y'
]

# Model architecture (same as training)
model = nn.Sequential(
    nn.Conv2d(1, 25, 3, stride=1, padding=1),
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
    nn.Linear(75 * 3 * 3, 512),
    nn.Dropout(0.4),
    nn.ReLU(),
    nn.Linear(512, len(classes))
).to(device)

# Load trained weights
model.load_state_dict(torch.load("best_sign_model.pth", map_location=device))
model.eval()


# Preprocess image
def preprocess_image(img):
    img = img.convert("L")   # grayscale
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)   # (1,28,28)
    img_array = np.expand_dims(img_array, axis=0)   # (1,1,28,28)
    return torch.tensor(img_array).float().to(device)

# Predict function
def predict_image(img):
    x = preprocess_image(img)
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted_index = np.argmax(probs)
        predicted_letter = classes[predicted_index]
        confidence = probs[predicted_index] * 100
    return predicted_letter, confidence

# Browse image
def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img_tk = ImageTk.PhotoImage(img.resize((200, 200)))
        panel.config(image=img_tk)
        panel.image = img_tk

        result, conf = predict_image(img)
        result_label.config(text=f"Prediction: {result} ({conf:.2f}%)")

# Capture webcam
def capture_webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("captured.png", gray)
        img = Image.open("captured.png")
        img_tk = ImageTk.PhotoImage(img.resize((200, 200)))
        panel.config(image=img_tk)
        panel.image = img_tk

        result, conf = predict_image(img)
        result_label.config(text=f"Prediction: {result} ({conf:.2f}%)")

# Tkinter GUI
root = tk.Tk()
root.title("Sign Language Alphabet Predictor (PyTorch)")

btn_file = tk.Button(root, text="Upload Image", command=browse_file)
btn_file.pack(pady=10)

btn_cam = tk.Button(root, text="Capture from Webcam", command=capture_webcam)
btn_cam.pack(pady=10)

panel = tk.Label(root)  # image display
panel.pack()

result_label = tk.Label(root, text="Prediction: None", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
