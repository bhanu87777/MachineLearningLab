import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("Iris.csv")

# Prepare features and target
X = df.drop(columns=["Id", "Species"])
y = df["Species"]

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build Random Forest with 10 trees
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

# Evaluate model
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Random Forest (10 trees) Accuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)

# Predict a new sample in one line output
new_sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
predicted_class = rf.predict(new_sample)[0]

print(f"\nInput {new_sample.values.tolist()[0]} classified as: {predicted_class}")
