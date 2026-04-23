from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Convert to binary classification (class 0 = positive, rest negative)
# You can change this condition based on your need
y_binary = [1 if label == 0 else 0 for label in y]

# Find-S Algorithm
def find_s(X, y):
    h = None
    
    for i in range(len(X)):
        if y[i] == 1:  # positive example
            if h is None:
                h = list(X[i])
            else:
                for j in range(len(h)):
                    if h[j] != X[i][j]:
                        h[j] = '?'
    
    return h

# Run algorithm
hypothesis = find_s(X, y_binary)

# Output
print("Final Hypothesis:")
print(hypothesis)
