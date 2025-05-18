# Build k-Means algorithm to cluster a set of data stored in a .CSV file. 

# You have to use: set OMP_NUM_THREADS=1 
# in anaconda prompt

import os
os.environ["OMP_NUM_THREADS"] = "1"  # Avoid Windows MKL memory leak warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_data(csv_path='Iris.csv'):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '') for c in df.columns]
        if 'species' not in df.columns and 'Species' not in df.columns:
            # Try to find a species column case-insensitive
            species_cols = [c for c in df.columns if 'species' in c.lower()]
            if species_cols:
                df['Species'] = df[species_cols[0]]
            else:
                raise ValueError("Species column not found in CSV")
        if 'Species' not in df.columns and 'species' in df.columns:
            df['Species'] = df['species']
    except Exception:
        # fallback to sklearn iris dataset
        iris = load_iris()
        df = pd.DataFrame(
            data=np.c_[iris['data'], iris['target']],
            columns=iris['feature_names'] + ['target']
        )
        df.columns = [c.strip().replace(' (cm)', '').replace(' ', '_') for c in df.columns]
        df['Species'] = [iris['target_names'][int(t)] for t in df['target']]
    return df

def preprocess(df):
    # Use PetalLengthCm and PetalWidthCm if present, else fall back to sklearn column names
    if 'PetalLengthCm' in df.columns and 'PetalWidthCm' in df.columns:
        X = df[['PetalLengthCm', 'PetalWidthCm']].values
    else:
        # Fall back to iris feature names without (cm)
        col_pl = next((c for c in df.columns if 'petal_length' in c.lower()), None)
        col_pw = next((c for c in df.columns if 'petal_width' in c.lower()), None)
        if col_pl and col_pw:
            X = df[[col_pl, col_pw]].values
        else:
            raise ValueError("Cannot find petal length and width columns.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def plot_elbow(X_scaled, max_k=10):
    inertias = []
    ks = range(1, max_k + 1)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertias, 'o-', linewidth=2)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(ks)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    return inertias

def run_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    return km, labels

def plot_confusion(df, labels, k):
    species_names = df['Species'].unique()
    species_to_num = {name: idx for idx, name in enumerate(species_names)}
    true_nums = df['Species'].map(species_to_num)
    cm = confusion_matrix(true_nums, labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[f"Cluster {i}" for i in range(k)])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_xlabel('Predicted Cluster')
    ax.set_ylabel('True Species')
    plt.title('K-Means Clustering Confusion Matrix')
    plt.tight_layout()
    plt.show()
    cm_df = pd.DataFrame(cm,
                         index=[f"True: {name}" for name in species_names],
                         columns=[f"Cluster {i}" for i in range(k)])
    print("\nConfusion Matrix (counts):")
    print(cm_df)

def main():
    df = load_data('Iris.csv')
    if 'Species' not in df.columns:
        print("Error: 'Species' column not found in the data.")
        return

    X_scaled, scaler = preprocess(df)

    print("Generating elbow plot to find optimal k...")
    plot_elbow(X_scaled, max_k=10)

    optimal_k = 3
    print(f"Choosing k = {optimal_k} based on elbow plot.")

    km_model, labels = run_kmeans(X_scaled, optimal_k)
    df['cluster'] = labels

    plt.figure(figsize=(6, 4))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    centroids = km_model.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', c='red', s=200, label='Centroids')
    plt.xlabel('Scaled Petal Length')
    plt.ylabel('Scaled Petal Width')
    plt.title(f'K-Means Clusters (k={optimal_k})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    plot_confusion(df, labels, optimal_k)

if __name__ == "__main__":
    main()
