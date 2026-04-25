import csv

def load_csv(filename):
    x, y = [], []

    with open(filename, newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)

        for row in reader:
            x.append(row[:-1])
            y.append(row[-1].strip().lower())
    
    return x, y

def find_s(x, y):
    h = None

    for i in range(len(x)):
        if y[i] in ('yes', '1', 'true', 'positive'):
            if h is None:
                h = list(x[i])
            else:
                for j in range(len(h)):
                    if h[j] != x[i][j]:
                        h[j] = '?'

    return h

x, y = load_csv('data.csv')

hypothesis = find_s(x, y)

print("Final Hypothesis:")
print(hypothesis)
