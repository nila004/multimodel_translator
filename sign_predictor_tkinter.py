import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("sign_language_model.h5")

# Correct classes mapping for Sign Language MNIST (24 classes, skipping J and Z)
classes = [
    'A','B','C','D','E','F','G','H','I',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y'
]

# Preprocess image for prediction (grayscale, 28x28x1)
def preprocess_image(img):
    img = img.convert("L")   # grayscale
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)   # (1,28,28)
    img_array = np.expand_dims(img_array, axis=-1)  # (1,28,28,1)
    return img_array

# Predict function
def predict_image(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_letter = classes[predicted_index]
    print("Prediction vector:", prediction)
    print("Predicted index:", predicted_index)
    print("Predicted letter:", predicted_letter)
    return predicted_letter

# Browse image from computer
def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img_tk = ImageTk.PhotoImage(img.resize((200, 200)))
        panel.config(image=img_tk)
        panel.image = img_tk

        result = predict_image(img)
        result_label.config(text=f"Prediction: {result}")

# Capture image from webcam
def capture_webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite("captured.png", frame)
        img = Image.open("captured.png")
        img_tk = ImageTk.PhotoImage(img.resize((200, 200)))
        panel.config(image=img_tk)
        panel.image = img_tk

        result = predict_image(img)
        result_label.config(text=f"Prediction: {result}")

# Tkinter GUI
root = tk.Tk()
root.title("Sign Language Alphabet Predictor")

btn_file = tk.Button(root, text="Upload Image", command=browse_file)
btn_file.pack(pady=10)

btn_cam = tk.Button(root, text="Capture from Webcam", command=capture_webcam)
btn_cam.pack(pady=10)

panel = tk.Label(root)  # image display
panel.pack()

result_label = tk.Label(root, text="Prediction: None", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
