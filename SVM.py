import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the IRIS dataset
iris_df = pd.read_csv("Iris.csv")

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

# Plot Confusion Matrix for Linear Kernel
plt.figure(figsize=(6, 4))
sns.heatmap(cm_linear, annot=True, fmt="d", cmap="Blues",
            xticklabels=svm_linear.classes_, yticklabels=svm_linear.classes_)
plt.title("Confusion Matrix - Linear Kernel")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

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

# Plot Confusion Matrix for RBF Kernel
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rbf, annot=True, fmt="d", cmap="Greens",
            xticklabels=svm_rbf.classes_, yticklabels=svm_rbf.classes_)
plt.title("Confusion Matrix - RBF Kernel")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
