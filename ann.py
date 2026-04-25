import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)

df = pd.read_csv("dataset.csv")
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

x = StandardScaler().fit_transform(x)
y = OneHotEncoder(sparse_output=False).fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr, epochs = 0.1, 1000

input, hidden, output = x.shape[1], 8, y.shape[1]

w1 = np.random.randn(input, hidden) * np.sqrt(1 / input)
w2 = np.random.randn(hidden, output) * np.sqrt(1 / hidden)

b1 = np.zeros((1, hidden))
b2 = np.zeros((1, output))

def sigmoid(x): return 1/(1 + np.exp(-x))

def sigmoid_grad(a): return a*(1-a)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e/e.sum(axis=1, keepdims=True)

for epoch in range(epochs):
    a1 = sigmoid(x_train @ w1 + b1)
    a2 = softmax(a1 @ w2 + b2)

    dz2 = (a2 - y_train) / len(x_train)
    dz1 = dz2 @ w2.T * sigmoid_grad(a1)

    w2 -= a1.T @ dz2 * lr; b2 -= dz2.sum(axis=0, keepdims=True) * lr
    w1 -= x_train.T @ dz1 * lr; b1 -= dz1.sum(axis=0, keepdims=True) * lr

def predict(x): return softmax(sigmoid(x @ w1 + b1) @ w2 + b2)

def accuracy(x, y): return np.mean(np.argmax(predict(x), axis=1) == np.argmax(y, axis=1))

print(f"Train: {accuracy(x_train, y_train) * 100:.2f} | Test: {accuracy(x_test, y_test) * 100:.2f}")
