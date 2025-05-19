import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Titanic dataset
df = pd.read_csv('Titanic.csv')

# Handle missing data
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert categorical variables to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Features and target
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Convert numeric output to binary labels
y_pred_labels = np.where(y_pred == 1, 'Yes', 'No')

# Optional: also convert actual y_test for easier comparison
y_test_labels = np.where(y_test == 1, 'Yes', 'No')

# Evaluation
print("Accuracy:", accuracy_score(y_test_labels, y_pred_labels))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_labels, y_pred_labels))
# Generate classification report as a dictionary
report = classification_report(y_test_labels, y_pred_labels, output_dict=True)

# Extract precision values
precision_no = report['No']['precision']
precision_yes = report['Yes']['precision']

# Print only the precision
print(f"Precision for 'No': {precision_no:.2f}")
print(f"Precision for 'Yes': {precision_yes:.2f}")
