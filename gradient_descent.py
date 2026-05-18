import pandas as pd

df = pd.read_csv("datasets/gb.csv")

x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

m, c = 0, 0
lr, epochs = 0.0001, 10000

n = len(x)

for _ in range(epochs):
    y_pred = m * x + c

    dm = (-2 / n) * sum(x * (y - y_pred))
    dc = (-2 / n) * sum(y - y_pred)

    m -= lr * dm
    c -= lr * dc

print(f"Slope (m): {m:.4f}")
print(f"Intercept (c): {c:.4f}")

x_new = float(input("Enter a value for x: "))
y_new = m * x_new + c

print(f"Predicted y: {y_new:.4f}")
