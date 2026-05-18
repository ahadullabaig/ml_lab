import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/km.csv")

x = df.iloc[:, :].values

k = 4

medoids = x[:k]

for _ in range(10):
    clusters = [[] for _ in range(k)]

    for i in x:
        distances = [np.linalg.norm(i - m) for m in medoids]
        clusters[np.argmin(distances)].append(i)

    new_medoids = []

    for c in clusters:
        c = np.array(c)
        costs = [np.sum(np.linalg.norm(c - p, axis=1)) for p in c]
        new_medoids.append(c[np.argmin(costs)])

    medoids = np.array(new_medoids)

for cluster in clusters:
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1])

plt.scatter(medoids[:, 0], medoids[:, 1], marker='x', s=200, c='black')
plt.title("K-Medoids Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
