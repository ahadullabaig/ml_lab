import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # Class label for leaf nodes

def entropy(y):
    probs = [np.mean(y == c) for c in np.unique(y)]
    return -sum(p * np.log2(p) for p in probs)

def information_gain(y, mask):
    parent_entropy = entropy(y)
    n = len(y)
    n_l, n_r = sum(mask), n - sum(mask)
    if n_l == 0 or n_r == 0: return 0
    child_entropy = (n_l/n) * entropy(y[mask]) + (n_r/n) * entropy(y[~mask])
    return parent_entropy - child_entropy

def build_tree(X, y, depth=0, max_depth=5):
    num_samples, num_features = X.shape
    if depth >= max_depth or len(np.unique(y)) == 1:
        return Node(value=np.argmax(np.bincount(y)))

    best_gain, best_split = -1, None
    for f in range(num_features):
        thresholds = np.unique(X[:, f])
        for t in thresholds:
            mask = X[:, f] <= t
            gain = information_gain(y, mask)
            if gain > best_gain:
                best_gain, best_split = gain, (f, t)

    f, t = best_split
    mask = X[:, f] <= t
    left = build_tree(X[mask], y[mask], depth + 1, max_depth)
    right = build_tree(X[~mask], y[~mask], depth + 1, max_depth)
    return Node(f, t, left, right)

def predict(node, x):
    if node.value is not None: return node.value
    if x[node.feature] <= node.threshold:
        return predict(node.left, x)
    return predict(node.right, x)

# --- Execution ---
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

tree = build_tree(X_train, y_train)
preds = [predict(tree, x) for x in X_test]
print(f"Accuracy: {np.mean(preds == y_test) * 100:.2f}%")