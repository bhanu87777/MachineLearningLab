# Build k-Means algorithm to cluster a set of data stored in a .CSV file. 

# You have to use: set OMP_NUM_THREADS=1 
# in anaconda prompt

import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data(csv_path='Iris.csv'):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '') for c in df.columns]
        if 'species' not in df.columns and 'Species' not in df.columns:
            species_cols = [c for c in df.columns if 'species' in c.lower()]
            if species_cols:
                df['Species'] = df[species_cols[0]]
            else:
                raise ValueError("Species column not found in CSV")
        if 'Species' not in df.columns and 'species' in df.columns:
            df['Species'] = df['species']
    except Exception:
        iris = load_iris()
        df = pd.DataFrame(
            data=np.c_[iris['data'], iris['target']],
            columns=iris['feature_names'] + ['target']
        )
        df.columns = [c.strip().replace(' (cm)', '').replace(' ', '_') for c in df.columns]
        df['Species'] = [iris['target_names'][int(t)] for t in df['target']]
    return df

def preprocess(df):
    if 'PetalLengthCm' in df.columns and 'PetalWidthCm' in df.columns:
        X = df[['PetalLengthCm', 'PetalWidthCm']].values
    else:
        col_pl = next((c for c in df.columns if 'petal_length' in c.lower()), None)
        col_pw = next((c for c in df.columns if 'petal_width' in c.lower()), None)
        if col_pl and col_pw:
            X = df[[col_pl, col_pw]].values
        else:
            raise ValueError("Cannot find petal length and width columns.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def run_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    return km, labels

def main():
    df = load_data('Iris.csv')
    if 'Species' not in df.columns:
        return

    X_scaled, scaler = preprocess(df)

    optimal_k = 3

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

if __name__ == "__main__":
    main()
