import matplotlib.pyplot as plt 

import pandas as pd 

from sklearn.preprocessing import StandardScaler 

from fcmeans import FCM 

from sklearn.datasets import load_wine 

  

# Load publicly available dataset (Wine dataset) 

wine = load_wine() 

X = wine.data[:, :2]  # Taking only the first two features for visualization 

  

# Standardizing the dataset 

scaler = StandardScaler() 

X_scaled = scaler.fit_transform(X) 

  

# Apply Fuzzy C-Means 

fcm = FCM(n_clusters=3, m=2.0, max_iter=150, error=1e-5, random_state=42) 

fcm.fit(X_scaled) 

  

# Get cluster centers and labels 

centers = fcm.centers 

labels = fcm.predict(X_scaled) 

  

# Plot results 

plt.figure(figsize=(8, 6)) 

for i in range(3): 

    plt.scatter(X_scaled[labels == i, 0], X_scaled[labels == i, 1], label=f'Cluster {i+1}') 

plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='black', s=200, label='Centers') 

plt.xlabel("Feature 1") 

plt.ylabel("Feature 2") 

plt.legend() 

plt.title("Fuzzy C-Means Clustering on Wine Dataset") 

plt.show() 