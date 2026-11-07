# Task 01 – Data Preparation and Baseline

## Project Overview
This project focuses on preparing a public dataset and building a simple Machine Learning classification model.

The objective is to establish a baseline using a Dummy Classifier and compare its performance with Logistic Regression.

## Dataset
- Dataset: Student Placement Dataset
- Source: [Kaggle – Factors Affecting Campus Placement](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)
- Problem Type: Binary Classification
- Target Column: `status`

The objective is to predict whether a student gets placed or not.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook/Visual Studio Code

## Project Workflow

1. Loaded the public dataset using Pandas.
2. Separated input features and target variable.
3. Removed the salary column to avoid data leakage.
4. Split the dataset into training (80%) and testing (20%) sets.
5. Created a baseline model using DummyClassifier with the most frequent strategy.
6. Applied preprocessing using StandardScaler and OneHotEncoder.
7. Trained a Logistic Regression model.
8. Evaluated both models using test accuracy.

## Model Performance

Model                         Test Accuracy
Dummy Classifier (Baseline)   69.77%
Logistic Regression           86.05%

Logistic Regression achieved an improvement of 16.28 percentage points over the baseline.

## Conclusion

The Logistic Regression model performed better than the trivial baseline.

The model achieved 86.05% test accuracy, demonstrating that the prepared dataset contains useful information for predicting student placement outcomes.

## How to Run the Project

1. Clone this repository:
   `git clone YOUR_GITHUB_REPOSITORY_URL`

2. Navigate to the project folder:
   `cd Task-01-Data-Preparation`

3. Install dependencies:
   `pip install -r requirements.txt`

4. Open Jupyter Notebook/Visual Studio Code:
   `jupyter notebook/Visual Studio Code`

5. Open and run the notebook inside the notebooks folder.

## Project Structure

    Task-01-Data-Preparation/
    ├── dataset/
    │   └── Placement_Data_Full_Class.csv
    ├── notebooks/
    │   └── task1.py
    ├── README.md
    └── requirements.txt

## Author
Machine Learning Internship – Task 01