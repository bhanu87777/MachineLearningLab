import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 1. Load sample data (Iris dataset)
data = load_iris()
X = data.data  # features
y = data.target  # labels (optional, for visualization)

# 2. Initialize PCA and reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. Show explained variance ratio by each component
print("Explained variance ratio:", pca.explained_variance_ratio_)

# 4. Plot the 2D transformed data
plt.figure(figsize=(8,6))
for target in np.unique(y):
    plt.scatter(
        X_pca[y == target, 0],
        X_pca[y == target, 1],
        label=data.target_names[target]
    )
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Iris dataset")
plt.legend()
plt.show()
