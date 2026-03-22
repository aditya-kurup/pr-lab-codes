import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def load_images(folder, img_size=(64, 64)):
    images, labels = [], []
    class_names = [c for c in os.listdir(folder) if os.path.isdir(os.path.join(folder, c))]

    for label, class_name in enumerate(class_names):
        for file in os.listdir(os.path.join(folder, class_name)):
            img = cv2.imread(os.path.join(folder, class_name, file))
            if img is not None:
                images.append(cv2.resize(img, img_size) / 255.0)
                labels.append(label)

    return np.array(images), np.array(labels), class_names

# Load and split data
folder_path = "path_to_your_images"
X, y, class_names = load_images(folder_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_cat = to_categorical(y_train, num_classes=len(class_names))
y_test_cat = to_categorical(y_test, num_classes=len(class_names))

# Build and train CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_cat, epochs=10, validation_data=(X_test, y_test_cat), batch_size=32)

loss, acc = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {acc:.2f}")