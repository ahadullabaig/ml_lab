import csv

def load_csv(filename):
    x, y = [], []

    with open(filename, newline='') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            x.append(row[:-1])
            y.append(row[-1].strip().lower())
    
    return x, y

def is_consistent(h, x):
    return all(h[i] == '?' or h[i] == x[i] for i in range(len(h)))

def generalize(h, x):
    result = []

    for i in range(len(h)):
        if h[i] == '0':
            result.append(x[i])
        elif h[i] != x[i]:
            result.append('?')
        else:
            result.append(h[i])
    
    return result

def candidate_elimination(x, y):
    n = len(x[0])
    S = [['0'] * n]
    G = [['?'] * n]

    for xi, label in zip(x, y):
        if label in ('yes', '1', 'true', 'positive'):
            G = [g for g in G if is_consistent(g, xi)]
            S = [generalize(s, xi) for s in S]
            S = [s for s in S if any(is_consistent(g, s) for g in G)]

        else:
            S = [s for s in S if not is_consistent(s, xi)]
            
            new_G = []

            for g in G:
                if is_consistent(g, xi):
                    for i in range(n):
                        if g[i] == '?':
                            for val in set(row[i] for row in x):
                                if val != xi[i]:
                                    h = g[:]
                                    h[i] = val
                                    if any(is_consistent(h, s) for s in S):
                                        new_G.append(h)
                else:
                    new_G.append(g)
            
            G = new_G

    return S, G

x, y = load_csv('data.csv')

S, G = candidate_elimination(x, y)

print("S (most specific):", S)
print("G (most general): ", G)
