import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset.csv")

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = np.hstack([np.ones((len(x_train), 1)), x_train])
x_test  = np.hstack([np.ones((len(x_test), 1)),  x_test])

w = np.linalg.pinv(x_train.T @ x_train) @ x_train.T @ y_train

y_pred = x_test @ w

mse  = np.mean((y_pred - y_test) ** 2)
rmse = np.sqrt(mse)
r2   = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")
