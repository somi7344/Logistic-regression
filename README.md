# Logistic-regression
End-to-end Logistic Regression project predicting stress levels using lifestyle data, with full data cleaning, preprocessing, model training, and business indights



````md
# Stress Level Prediction using Logistic Regression

---

## Table of Contents
- [Project Overview](#project-overview)
- [Objective](#objective)
- [Dataset Columns](#dataset-columns)
- [Step-by-Step Process](#step-by-step-process-with-code)
  - [Import Libraries](#1-import-libraries)
  - [Load Dataset](#2-load-dataset)
  - [Check for Missing Values](#3-check-for-missing-values)
  - [Convert Categorical Data](#4-convert-categorical-data)
  - [Convert Target Variable](#5-convert-target-variable)
  - [Split Features and Target](#6-split-features-and-target)
  - [Train-Test Split](#7-train-test-split)
  - [Train Model](#8-train-logistic-regression-model)
  - [Make Predictions](#9-make-predictions)
  - [Evaluate Model](#10-evaluate-model)
- [Results and Interpretation](#results-and-interpretation)
- [Key Insights](#key-insights)
- [Example Prediction](#example-prediction)
- [Real-World Applications](#real-world-applications)
- [Conclusion](#conclusion)
- [Future Improvements](#future-improvements)
- [Tools Used](#tools-used)
- [Author](#author)

---

## Project Overview
This project predicts whether an individual is likely to experience high stress or low stress based on lifestyle factors such as sleep, working hours, exercise frequency, and job type.

---

## Objective
- Classify individuals into High Stress (1) or Low Stress (0)
- Identify key lifestyle factors affecting stress
- Build a predictive model

---

## Dataset Columns
- Age  
- Working_Hours  
- Sleep_Hours  
- Exercise_Freq  
- Job_Type  
- Stress_Level  

---

## Step-by-Step Process (With Code)

---

## 1. Import Libraries
```python
import pandas as pd
import numpy as np
---

## 2. Load Dataset

```python
df = pd.read_csv('data.csv')
df.head()
```

Loads the dataset and displays the first few rows.

---

## 3. Check for Missing Values

```python
df.isnull().sum()
```

If missing values exist:

```python
df = df.dropna()
```

Removes incomplete data to ensure accuracy.

---

## 4. Convert Categorical Data

```python
df = pd.get_dummies(df, drop_first=True)
```

Converts categorical variables into numerical format using 0s and 1s.

---

## 5. Convert Target Variable

```python
df['Stress_Level'] = df['Stress_Level'].map({'Low': 0, 'High': 1})
```

Transforms the target variable into numerical form.

---

## 6. Split Features and Target

```python
X = df.drop('Stress_Level', axis=1)
y = df['Stress_Level']
```

---

## 7. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 8. Train Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

---

## 9. Make Predictions

```python
y_pred = model.predict(X_test)
```

---

## 10. Evaluate Model

```python
from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## Results and Interpretation

The model achieved good performance, indicating that lifestyle factors significantly influence stress levels.

---

## Key Insights

* Low sleep hours are associated with higher stress
* Longer working hours increase stress probability
* Regular exercise is linked to lower stress levels
* Certain job types show higher stress patterns

---

## Example Prediction

```python
model.predict([[30, 10, 4, 0]])
```

---

## Real-World Applications

* Organizations can monitor employee well-being
* Healthcare systems can support early stress detection
* Applications can be developed for stress tracking

---

## Conclusion

This project demonstrates how machine learning can be used to analyze lifestyle data and predict stress levels.

---

## Future Improvements

* Use larger datasets
* Apply advanced models
* Add more features
* Improve accuracy

---

## Tools Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## Author

Chisom
Aspiring Data Scientist and Machine Learning Engineer

