import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("adult.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate features and target
X = df.drop(columns=['income'], errors='ignore', axis=1)
y = df['income']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base estimator
base_est = DecisionTreeClassifier(max_depth=3)

# AdaBoost with 10 estimators using SAMME algorithm
model_10 = AdaBoostClassifier(estimator=base_est, n_estimators=10, random_state=42, algorithm='SAMME')
model_10.fit(X_train, y_train)
y_pred_10 = model_10.predict(X_test)
score_10 = accuracy_score(y_test, y_pred_10)
print(f"Accuracy with 10 estimators: {score_10:.4f}")
print("Classification Report (10 Estimators):\n", classification_report(y_test, y_pred_10))

# Initialize variables for fine-tuning
estimators_range = list(range(10, 201, 10))
scores = []
best_score = 0
best_n = 0

# Fine-tune number of estimators
for n in estimators_range:
    model = AdaBoostClassifier(estimator=base_est, n_estimators=n, random_state=42, algorithm='SAMME')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
    print(f"n_estimators={n}, Accuracy={score:.4f}")
    if score > best_score:
        best_score = score
        best_n = n
        best_model = model  # Save the best model
        best_y_pred = y_pred

print(f"\nBest Accuracy: {best_score:.4f} using {best_n} estimators")
print("Classification Report (Best Estimator):\n", classification_report(y_test, best_y_pred))

# Plot accuracy vs number of estimators
plt.figure(figsize=(7, 4))
plt.plot(estimators_range, scores, marker='o', linestyle='-', color='blue')
plt.title("Accuracy vs Number of Estimators (AdaBoost)")
plt.xlabel("Number of Estimators (Trees)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.xticks(estimators_range)
plt.tight_layout()
plt.show()

# Visualize feature importances for best model
importances = best_model.feature_importances_
features = X.columns
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances (AdaBoost)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
