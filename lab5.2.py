import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_images(folder, img_size=(64, 64)):
    images, class_names = [], []
    class_names = [c for c in os.listdir(folder) if os.path.isdir(os.path.join(folder, c))]

    for class_name in class_names:
        for file in os.listdir(os.path.join(folder, class_name)):
            img = cv2.imread(os.path.join(folder, class_name, file))
            if img is not None:
                images.append(cv2.resize(img, img_size) / 255.0)

    return np.array(images), class_names

# Load data
folder_path = "path_to_your_images"
X, class_names = load_images(folder_path)

# Flatten and cluster
X_flat = X.reshape(X.shape[0], -1)
kmeans = KMeans(n_clusters=len(class_names), random_state=42)
labels = kmeans.fit_predict(X_flat)

# Plot results
fig, axes = plt.subplots(1, 5, figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.imshow(X[i])
    ax.set_title(f"Cluster {labels[i]}")
    ax.axis('off')
plt.show()