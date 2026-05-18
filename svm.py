import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df = pd.read_csv("datasets/svm.csv")

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = SVC(kernel="linear")
model.fit(x_train, y_train)

print(f"Accuracy: {model.score(x_test, y_test) * 100:.2f}%")

for c in np.unique(y):
    plt.scatter(x[y == c, 0], x[y == c, 1], label=f"Class {c}")

w = model.coef_[0]
b = model.intercept_[0]

xs = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
ys = -(w[0] * xs + b) / w[1]

plt.plot(xs, ys, 'k--', label="Decision boundary")
plt.title("SVM")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
