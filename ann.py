import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

iris  = load_iris()
X     = iris.data / iris.data.max(axis=0)
y     = OneHotEncoder(sparse_output=False).fit_transform(iris.target.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr, epochs = 0.1, 1000
input_sz, hidden_sz, output_sz = 4, 8, 3

W1 = np.random.randn(input_sz,  hidden_sz) * np.sqrt(1 / input_sz)
W2 = np.random.randn(hidden_sz, output_sz) * np.sqrt(1 / hidden_sz)
b1 = np.zeros((1, hidden_sz))
b2 = np.zeros((1, output_sz))

def sigmoid(x):      return 1 / (1 + np.exp(-x))
def sigmoid_grad(a): return a * (1 - a)
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

for epoch in range(epochs):
    a1 = sigmoid(X_train @ W1 + b1)
    a2 = softmax(a1     @ W2 + b2)

    dz2 = (a2 - y_train) / len(X_train)
    dz1 = (dz2 @ W2.T) * sigmoid_grad(a1)

    W2 -= a1.T      @ dz2 * lr;  b2 -= dz2.sum(axis=0, keepdims=True) * lr
    W1 -= X_train.T @ dz1 * lr;  b1 -= dz1.sum(axis=0, keepdims=True) * lr

    if epoch % 200 == 0:
        loss = -np.mean(np.sum(y_train * np.log(a2.clip(1e-9)), axis=1))
        print(f"Epoch {epoch:4d}  loss: {loss:.4f}")

def predict(X):  return softmax(sigmoid(X @ W1 + b1) @ W2 + b2)
def accuracy(X, y): return np.mean(np.argmax(predict(X), axis=1) == np.argmax(y, axis=1))

print(f"Train: {accuracy(X_train, y_train)*100:.1f}%  |  Test: {accuracy(X_test, y_test)*100:.1f}%")
