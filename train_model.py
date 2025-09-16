import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data
train_df = pd.read_csv("data/sign_mnist_train.csv")
test_df = pd.read_csv("data/sign_mnist_test.csv")

# Separate labels and images
y_train = train_df['label'].values
X_train = train_df.drop('label', axis=1).values
y_test = test_df['label'].values
X_test = test_df.drop('label', axis=1).values

# ✅ Fix labels (remove J=9 and Z=25, reindex others)
def fix_labels(y):
    y_fixed = []
    for label in y:
        if label == 9:  
            continue
        elif label == 25: 
            continue
        elif label > 9 and label < 25:
            y_fixed.append(label - 1)
        elif label > 25:
            y_fixed.append(label - 2)
        else:
            y_fixed.append(label)
    return np.array(y_fixed)

y_train = fix_labels(y_train)
y_test = fix_labels(y_test)

# Normalize and reshape
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# One-hot encode labels
num_classes = len(np.unique(y_train))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=35, batch_size=64,
                    validation_data=(X_test, y_test))

# Save model
model.save("sign_language_model.h5")
print("✅ Model trained and saved as sign_language_model.h5")
