import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("datasets/logisticreg.csv")

x = df.iloc[:, 0].values / 100
y = df.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def sigmoid(z): return 1 / (1 + np.exp(-z))

w, b = 0.0, 0.0
lr, epochs = 0.5, 5000

for _ in range(epochs):
    p = sigmoid(w * x_train + b)

    dw = np.mean((p - y_train) * x_train)
    db = np.mean(p - y_train)

    w -= lr * dw
    b -= lr * db

preds    = (sigmoid(w * x_test + b) >= 0.5).astype(int)
accuracy = np.mean(preds == y_test) * 100

print(f"Weight: {w:.4f}, Bias: {b:.4f}")
print(f"Accuracy: {accuracy:.2f}%")

plt.plot(y_test, label="Actual", marker="o")
plt.plot(preds, label="Predicted", marker="x", linestyle="--")
plt.legend()
plt.show()
