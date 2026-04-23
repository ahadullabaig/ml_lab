import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# Load and preprocess data
iris = load_iris()
X = iris.data / np.max(iris.data, axis=0) # Simple normalization
y = iris.target.reshape(-1, 1)
y = OneHotEncoder(sparse_output=False).fit_transform(y) # One-hot encoding

# Hyperparameters
input_size, hidden_size, output_size = 4, 8, 3
lr = 0.1
epochs = 1000

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_der(x): return x * (1 - x)

# Training Loop
for _ in range(epochs):
    # Forward Pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2) # Output layer

    # Backpropagation
    error = y - a2
    d_a2 = error * sigmoid_der(a2)
    
    error_h = d_a2.dot(W2.T)
    d_a1 = error_h * sigmoid_der(a1)

    # Update Weights/Biases
    W2 += a1.T.dot(d_a2) * lr
    b2 += np.sum(d_a2, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_a1) * lr
    b1 += np.sum(d_a1, axis=0, keepdims=True) * lr

# Final Prediction
final_out = sigmoid(np.dot(sigmoid(np.dot(X, W1) + b1), W2) + b2)
accuracy = np.mean(np.argmax(final_out, axis=1) == np.argmax(y, axis=1))
print(f"Accuracy: {accuracy * 100:.2f}%")