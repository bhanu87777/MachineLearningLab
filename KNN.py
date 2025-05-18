import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load iris dataset
iris_df = pd.read_csv("Iris.csv")

# Drop the Id column if it exists
if 'Id' in iris_df.columns:
    iris_df = iris_df.drop('Id', axis=1)

# Features and target
X_iris = iris_df.drop('Species', axis=1)
y_iris = iris_df['Species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN
knn_iris = KNeighborsClassifier(n_neighbors=3)
knn_iris.fit(X_train, y_train)

# Predict
y_pred = knn_iris.predict(X_test)

# Evaluate
print("Iris Dataset - Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt='g')
plt.title("Confusion Matrix - Iris KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
