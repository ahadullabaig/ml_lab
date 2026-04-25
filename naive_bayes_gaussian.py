import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset.csv")

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

classes = np.unique(y_train)

mean = np.array([x_train[y_train == c].mean(axis=0) for c in classes])
var  = np.array([x_train[y_train == c].var(axis=0)  for c in classes])
prior = np.array([np.mean(y_train == c) for c in classes])

def gaussian(x, mean, var):
    return np.exp(-0.5 * ((x - mean) ** 2) / var) / np.sqrt(2 * np.pi * var)

def predict(x):
    posteriors = []

    for i, c in enumerate(classes):
        likelihood = np.prod(gaussian(x, mean[i], var[i]), axis=1)
        posteriors.append(likelihood * prior[i])
    
    return classes[np.argmax(posteriors, axis=0)]

preds    = predict(x_test)
accuracy = np.mean(preds == y_test) * 100

print(f"Accuracy: {accuracy:.2f}%")
