import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Libraries imported successfully!")
iris = load_iris()

print("Iris dataset loaded successfully!")
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

print(df.head())
df["species_name"] = df["species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

print(df.head())
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSpecies Count:")
print(df["species_name"].value_counts())
df["species_name"].value_counts().plot(kind="bar")

plt.title("Iris Flower Species Distribution")
plt.xlabel("Species")
plt.ylabel("Number of Flowers")

plt.show()
X = df[
    [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
]

y = df["species"]

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed!")


model = LogisticRegression(max_iter=200)

model.fit(X_train_scaled, y_train)

print("Model training completed!")

y_pred = model.predict(X_test_scaled)

print("Predictions:")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print(
    "Accuracy Percentage:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1, 2],
    iris.target_names
)

plt.yticks(
    [0, 1, 2],
    iris.target_names
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.show()

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

print(
    "\nPredicted Flower Species:",
    iris.target_names[prediction[0]]
)
