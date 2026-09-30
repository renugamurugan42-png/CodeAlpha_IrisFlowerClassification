# 🌸 Iris Flower Classification

## 📌 Project Overview

This project is a Machine Learning-based Iris Flower Classification system developed as part of the CodeAlpha Internship.

The system predicts the species of an Iris flower based on its sepal and petal measurements.

## 🎯 Objective

The main objective of this project is to build a machine learning model that can classify Iris flowers into three different species:

* Setosa
* Versicolor
* Virginica

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Logistic Regression
* VS Code

## 📊 Dataset

The Iris dataset is obtained using Scikit-learn's built-in `load_iris()` dataset.

The dataset contains 150 flower samples with four features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

There are three target classes:

* Setosa
* Versicolor
* Virginica

## ⚙️ Methodology

The project follows these steps:

1. Import the required Python libraries.
2. Load the Iris dataset.
3. Convert the dataset into a Pandas DataFrame.
4. Check the dataset information and missing values.
5. Visualize the species distribution.
6. Separate features and target values.
7. Split the dataset into training and testing data.
8. Apply feature scaling using StandardScaler.
9. Train a Logistic Regression classification model.
10. Predict the flower species using the test data.
11. Evaluate the model using accuracy and classification report.
12. Generate a confusion matrix.
13. Predict the species of a new Iris flower using its measurements.

## 🤖 Machine Learning Algorithm

### Logistic Regression

Logistic Regression is used as the classification algorithm to predict the species of Iris flowers based on their measurements.

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The actual accuracy obtained during execution is reported from the model output.

## 🌸 Sample Prediction

A new flower sample is provided with the following measurements:

* Sepal Length: 5.1 cm
* Sepal Width: 3.5 cm
* Petal Length: 1.4 cm
* Petal Width: 0.2 cm

The trained model predicts the flower species based on these measurements.

## 📁 Project Structure

```text
CodeAlpha_IrisFlowerClassification
│
├── iris_classification.py
└── README.md
```

## 👩‍💻 Internship

This project was completed as **Task 1 – Iris Flower Classification** as part of the CodeAlpha Internship Program.

## 👤 Author

RENUGA M

B.Tech Artificial Intelligence and Data Science
