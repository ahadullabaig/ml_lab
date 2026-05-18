import pandas as pd
import matplotlib.pyplot as plt

f = pd.read_csv("datasets/gb.csv")

x = f.iloc[:, 0].values
y = f.iloc[:, 1].values

m, c = 0, 0

lr, epochs = 0.0001, 10000

n = len(x)

for _ in range(epochs):
    y_pred = m*x + c

    dm = (-2 / n) * sum(x * (y - y_pred))
    dc = (-2 / n) * sum(y - y_pred)

    m -= lr * dm
    c -= lr * dc

print(f"Slope (m): {m:.4f}")
print(f"Intercept (c): {c:.4f}")

plt.scatter(x, y)

plt.plot(x, m*x+c)

plt.xlabel('x')
plt.ylabel('y')

plt.show()
