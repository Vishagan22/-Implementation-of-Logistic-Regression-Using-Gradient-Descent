# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Step 2: Load Data
df = pd.read_csv('Placement_Data-3.csv')

# Step 3: Drop irrelevant columns (keep only useful features for classification)
df = df.drop(['slno', 'salary'], axis=1, errors='ignore')

# Step 4: Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Step 5: Define X and y
X = df.drop('status', axis=1).values
y = df['status'].values  # 1 means 'Placed', 0 means 'Not Placed' (after encoding)

# Step 6: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 7: Add bias column to X
def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

X_train_b = add_bias(X_train)
X_test_b = add_bias(X_test)

# Step 8: Sigmoid and loss function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, w):
    m = X.shape[0]
    h = sigmoid(X @ w)
    epsilon = 1e-7  # to avoid log(0)
    loss = -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return loss

# Step 9: Gradient Descent
def gradient_descent(X, y, learning_rate=0.01, n_iters=1000):
    m, n = X.shape
    w = np.zeros(n)
    for _ in range(n_iters):
        h = sigmoid(X @ w)
        grad = (1/m) * (X.T @ (h - y))
        w -= learning_rate * grad
    return w

# Step 10: Train Model
weights = gradient_descent(X_train_b, y_train, learning_rate=0.1, n_iters=5000)

# Step 11: Predictions
y_pred_prob = sigmoid(X_test_b @ weights)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Step 12: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

```

## Output:

<img width="907" height="263" alt="image" src="https://github.com/user-attachments/assets/0a94ab03-54ee-4e69-9b6b-8c33dfd5db42" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

