import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

x = df.iloc[:, 0]
y = df.iloc[:, 1]

x_mean = x.mean()
y_mean = y.mean()

num = ((x - x_mean) * (y - y_mean)).sum()
den = ((x - x_mean) ** 2).sum()

m = num / den

c = y_mean - m * x_mean

print("Slope:", m)
print("Intercept:", c)

y_pred = m*x + c

print("Predicted values:\n", y_pred)

plt.scatter(x, y, label="Actual data")
plt.plot(x, y_pred, label="Regression line")
plt.legend()
plt.show()

# x_new = float(input("Enter a value for x: "))
# y_new = m*x_new + c

# print("Predicted value of y:", y_new)
