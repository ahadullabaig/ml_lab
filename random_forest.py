import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("datasets/randomforest.csv")

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

print(f"Accuracy: {model.score(x_test, y_test) * 100:.2f}%")

print("\nFeature importances:")
for name, imp in zip(x.columns, model.feature_importances_):
    print(f"  {name}: {imp:.4f}")
