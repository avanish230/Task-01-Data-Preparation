import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("dataset/Placement_Data_Full_Class.csv")

print(data.head())
print(data.shape)
print(data.info())
X = data.drop(columns=["status", "salary"])
y = data["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_pred)

print("Baseline Accuracy:", baseline_accuracy)

numeric_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_columns),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

model_accuracy = accuracy_score(y_test, predictions)

print("Model Evaluation Summary")
print("------------------------")
print("Baseline Accuracy:", baseline_accuracy)
print("Logistic Regression Accuracy:", model_accuracy)

if model_accuracy > baseline_accuracy:
    print("Result: Model beats the baseline")
else:
    print("Result: Model does not beat the baseline")

print(classification_report(y_test, predictions))

