import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = pd.read_csv("datasets/km.csv").values

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

for i, c in enumerate(clusters):
    c = np.array(c)
    plt.scatter(c[:, 0], c[:, 1])

plt.scatter(medoids[:, 0], medoids[:, 1], marker='x', s=200, c='black')
plt.show()
