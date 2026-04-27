import pandas as pd

df = pd.read_csv("dataset.csv")

x = df.iloc[:, 0]
y = df.iloc[:, 1]

x_mean = x.mean()
y_mean = y.mean()

num = ((x - x_mean) * (y - y_mean)).sum()
den = ((x - x_mean) ** 2).sum()

b1 = num / den

b0 = y_mean - b1 * x_mean

print("Slope (b1):", b1)
print("Intercept (b0):", b0)

y_pred = b0 + b1 * x

print("Predicted values:\n", y_pred)

x_new = float(input("Enter a value for x: "))
y_new = b0 + b1 * x_new

print("Predicted value of y:", y_new)
