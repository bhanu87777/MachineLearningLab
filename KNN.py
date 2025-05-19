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

# Predict test set
y_pred = knn_iris.predict(X_test)

# Evaluate
print("Iris Dataset - Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict for new data entry
# Predict for new data entry with feature names
new_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X_iris.columns)
new_data_scaled = scaler.transform(new_data)
predicted_class = knn_iris.predict(new_data_scaled)

print(f"\nThe new data entry {new_data.values.tolist()} is classified as: {predicted_class[0]}")
