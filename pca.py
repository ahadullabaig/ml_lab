import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv("datasets/pca.csv")

x = df.iloc[:, :-1]

print("Initial Shape:", x.shape)
print(df)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

print("Final Shape:", x_pca.shape)
