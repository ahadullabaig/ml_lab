import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset.csv")

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(x_train, y_train)

print(f"Accuracy: {model.score(x_test, y_test) * 100:.2f}%")

plt.figure(figsize=(14, 7))
plot_tree(model, feature_names=list(x.columns), class_names=model.classes_.astype(str), filled=True)
plt.show()
