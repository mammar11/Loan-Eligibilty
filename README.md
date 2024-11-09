# Loan Eligibility Prediction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Loan Prediction](https://img.shields.io/badge/Loan-Eligibility-green.svg)

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Preprocessing](#preprocessing)
  - [Feature Selection and Extraction](#feature-selection-and-extraction)
  - [Encoding and Normalization](#encoding-and-normalization)
- [Classification Models](#classification-models)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

## Introduction

This project predicts loan eligibility based on several applicant features, using machine learning classification algorithms. It aims to assist financial institutions in making efficient and accurate decisions regarding loan approvals.

## Project Overview

The project follows these main steps:

1. **Exploratory Data Analysis (EDA)**: Understanding data patterns, distributions, and correlations.
2. **Preprocessing**:
   - **Feature Selection and Extraction**: Converting categorical variables, applying One Hot Encoding, and normalizing the data.
3. **Classification**: Applying multiple machine learning algorithms to classify applicants.
4. **Model Evaluation**: Using metrics such as F1 score, Jaccard index, and Log Loss to evaluate model performance.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - **Data Analysis**: Pandas, NumPy
  - **Visualization**: Matplotlib, Seaborn
  - **Machine Learning**: Scikit-Learn
  - **Metrics**: Scikit-Learn metrics (F1 score, Jaccard, Log Loss)

## Exploratory Data Analysis (EDA)

The EDA step involved understanding data distribution, checking for missing values, and identifying patterns and correlations in the features. Key insights from EDA helped guide the preprocessing and feature engineering steps.

### Sample EDA Code

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('loan_data.csv')

# Check for missing values
missing_values = data.isnull().sum()

# Distribution of target variable
sns.countplot(x='Loan_Status', data=data)
plt.title("Loan Status Distribution")
plt.show()
```
## Preprocessing
### Feature Selection and Extraction
This step involves preparing the data by converting categorical features to numerical values, selecting important features, and applying One Hot Encoding.
### Encoding and Normalization
- Categorical to Numerical Conversion: Converting text features (e.g., gender, education) to numerical values.
- One Hot Encoding: Used to convert categorical variables into binary columns.
- Normalization: Scaling the data to improve model performance.
```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# One Hot Encoding categorical variables
data = pd.get_dummies(data, columns=['Gender', 'Education'], drop_first=True)

# Normalize numeric features
scaler = StandardScaler()
data[['ApplicantIncome', 'LoanAmount']] = scaler.fit_transform(data[['ApplicantIncome', 'LoanAmount']])
```
## Classification Models
Several machine learning algorithms were used to predict loan eligibility:

- K-Nearest Neighbors (KNN): A simple yet effective algorithm for binary classification.
- Decision Tree: A tree-based model that captures non-linear relationships.
- Support Vector Machine (SVM): A model that maximizes the margin between classes.
- Logistic Regression: A linear model useful for binary classification problems
```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
svm = SVC()
lr = LogisticRegression()

# Training each model
knn.fit(X_train, y_train)
dt.fit(X_train, y_train)
svm.fit(X_train, y_train)
lr.fit(X_train, y_train)
```
## Model Evaluation
The models were evaluated using the following metrics:

- F1 Score: Balances precision and recall.
- Jaccard Index: Measures the similarity between predicted and actual results.
- Log Loss: Penalizes incorrect predictions, with a focus on probabilistic outputs.
```python
from sklearn.metrics import f1_score, jaccard_score, log_loss

# Predictions
y_pred_knn = knn.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_lr = lr.predict(X_test)

# Evaluation
print("KNN F1 Score:", f1_score(y_test, y_pred_knn))
print("Decision Tree Jaccard Score:", jaccard_score(y_test, y_pred_dt))
print("SVM Log Loss:", log_loss(y_test, svm.predict_proba(X_test)))
print("Logistic Regression F1 Score:", f1_score(y_test, y_pred_lr))
```
## Results
The results indicate the accuracy and effectiveness of each classification model. Below is a summary of model performance based on F1 score, Jaccard index, and Log Loss.
| Model | f1-score | Jaccard | logloss |
|--------------|-----------|-----------|-----------|
| K-Nearest Neighbors	| 0.6468253968253967	| 0.6468253968253967	| NA |
| Decision Tree |	0.6304176516942475		| 0.6304176516942475	| NA |
| SVM	| 0.6378600823045267	| 0.6378600823045267	| NA |
| Logistic Regression |	0.6304176516942475 |	0.6304176516942475 | 0.5164726137314147 |
## Conclusion
The project demonstrates effective preprocessing, model training, and evaluation for loan eligibility prediction. Logistic Regression provided [interpret results here, e.g., the highest accuracy, balance, etc.].
## Project Structure
```css
loan-eligibility-prediction/
│
├── data/
│   ├── loan_data.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model_training.py
│
├── README.md
└── LICENSE
```
## License
This project is licensed under the Apache-2.0 License. See the LICENSE file for details.
## Contact
Mohammed Ammaruddin
md.ammaruddin2020@gmail.com
