import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode

df = pd.read_csv("dataset.csv")

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x = StandardScaler().fit_transform(x)

k = len(np.unique(y))

np.random.seed(42)

centroids = x[np.random.choice(len(x), k, replace=False)]

for _ in range(100):
    distances  = np.linalg.norm(x[:, None] - centroids, axis=2)
    clusters   = np.argmin(distances, axis=1)

    new_centroids = np.array([x[clusters == i].mean(axis=0) for i in range(k)])

    if np.allclose(centroids, new_centroids): break

    centroids = new_centroids

labels = np.zeros_like(clusters)

for i in range(k):
    mask = clusters == i
    labels[mask] = mode(y[mask], keepdims=True)[0][0]

accuracy = np.mean(labels == y) * 100

print(f"Accuracy: {accuracy:.2f}%")

plt.figure(figsize=(8, 5))
plt.scatter(x[:, 0], x[:, 1], c=clusters, cmap="viridis")
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroids")
plt.title("K-Means Clustering")
plt.legend()
plt.show()
