import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

x = df.iloc[:, :2].values 
# change to df.iloc[:, :-1].values if more features

k = 3
# change to len(np.unique(df.iloc[:, -1])) if different classes

centroids = x[:k]

for _ in range(10):
    clusters = [[] for _ in range(k)]

    for i in x:
        distances = [np.linalg.norm(i - c) for c in centroids]
        clusters[np.argmin(distances)].append(i)

    new_centroids = []
    
    for c in clusters:
        if len(c) == 0:
            new_centroids.append(x[np.random.randint(0, len(x))])
        else:
            new_centroids.append(np.mean(c, axis=0))

    centroids = np.array(new_centroids)

for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1])

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
