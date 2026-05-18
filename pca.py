import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("datasets/pca.csv")

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

print(f"Explained variance: {pca.explained_variance_ratio_}")
print(f"Total: {pca.explained_variance_ratio_.sum() * 100:.2f}%")

for c in np.unique(y):
    plt.scatter(x_pca[y == c, 0], x_pca[y == c, 1], label=f"Class {c}")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA")
plt.legend()
plt.show()
