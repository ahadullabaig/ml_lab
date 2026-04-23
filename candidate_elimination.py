import csv

def load_csv(filename):
    X, y = [], []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            X.append(row[:-1])
            y.append(row[-1].strip().lower())
    return X, y

def is_consistent(h, x):
    return all(h[i] == '?' or h[i] == x[i] for i in range(len(h)))

def generalize(h, x):
    return [x[i] if h[i] == '0' else ('?' if h[i] != x[i] else h[i]) for i in range(len(h))]

def candidate_elimination(X, y):
    n = len(X[0])
    S = [['0'] * n]          # most specific
    G = [['?'] * n]          # most general

    for x, label in zip(X, y):
        if label in ('yes', '1', 'true', 'positive'):
            G = [g for g in G if is_consistent(g, x)]
            S = [generalize(s, x) for s in S]
            S = [s for s in S if any(is_consistent(g, s) for g in G)]
        else:
            S = [s for s in S if not is_consistent(s, x)]
            new_G = []
            for g in G:
                if is_consistent(g, x):
                    for i in range(n):
                        if g[i] == '?':
                            for val in set(row[i] for row in X):
                                if val != x[i]:
                                    h = g[:]
                                    h[i] = val
                                    if any(is_consistent(h, s) for s in S):
                                        new_G.append(h)
                else:
                    new_G.append(g)
            G = new_G

    return S, G

X, y = load_csv('data.csv')
S, G = candidate_elimination(X, y)

print("S (most specific):", S)
print("G (most general): ", G)
