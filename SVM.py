import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the IRIS dataset
iris_df = pd.read_csv("Iris.csv")

# Drop the 'Id' column if it exists
if 'Id' in iris_df.columns:
    iris_df = iris_df.drop('Id', axis=1)

# Split into features and target
X = iris_df.drop("Species", axis=1)
y = iris_df["Species"]

# Split into training and testing (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- SVM with Linear Kernel ---
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

# Accuracy and Confusion Matrix
acc_linear = accuracy_score(y_test, y_pred_linear)
cm_linear = confusion_matrix(y_test, y_pred_linear)

print("Linear Kernel:")
print("Accuracy:", acc_linear)
print("Confusion Matrix:\n", cm_linear)

# --- SVM with RBF Kernel ---
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

# Accuracy and Confusion Matrix
acc_rbf = accuracy_score(y_test, y_pred_rbf)
cm_rbf = confusion_matrix(y_test, y_pred_rbf)

print("\nRBF Kernel:")
print("Accuracy:", acc_rbf)
print("Confusion Matrix:\n", cm_rbf)


# --- Predict for a new data entry ---
# Replace these values with any new sample features
# Your feature names (check what columns you trained the model on)
feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# New sample as a DataFrame with same column names
new_sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=feature_names)

# Predict with Linear kernel SVM
predicted_class_linear = svm_linear.predict(new_sample)
print(f"\nNew data entry {new_sample} classified by Linear Kernel SVM as: {predicted_class_linear[0]}")

# Predict with RBF kernel SVM
predicted_class_rbf = svm_rbf.predict(new_sample)
print(f"New data entry {new_sample} classified by RBF Kernel SVM as: {predicted_class_rbf[0]}")
