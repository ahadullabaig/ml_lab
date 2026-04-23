import csv

# Load dataset from CSV (last column = target label)
def load_csv(filename):
    X, y = [], []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)  # skip header row
        for row in reader:
            X.append(row[:-1])
            y.append(row[-1].strip().lower())
    return X, y

# Find-S Algorithm
def find_s(X, y):
    h = None

    for i in range(len(X)):
        if y[i] in ('yes', '1', 'true', 'positive'):  # positive example
            if h is None:
                h = list(X[i])
            else:
                for j in range(len(h)):
                    if h[j] != X[i][j]:
                        h[j] = '?'

    return h

# Run algorithm
X, y = load_csv('data.csv')
hypothesis = find_s(X, y)

# Output
print("Final Hypothesis:")
print(hypothesis)
