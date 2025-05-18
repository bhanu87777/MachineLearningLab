import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats

# Diabetes Dataset
df = pd.read_csv('diabetes.csv')

# Loading and Inspecting the Dataset
df.head()
df.shape
print(df.info())
print(df.describe())

# Checking for Missing Values
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Identifying and Encoding Categorical Columns
categorical_cols = df.select_dtypes(include=['object']).columns
print("Categorical columns identified:", categorical_cols)

if len(categorical_cols) > 0:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print("\nDataFrame after one-hot encoding:")
    print(df.head())
else:
    print("\nNo categorical columns found in the dataset.")

# Feature Scaling
numerical_cols = df.select_dtypes(include=['number']).columns

# a. Min-Max Scaling
scaler = MinMaxScaler()
df_minmax = df.copy()
df_minmax[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# b. Standardization (Z-score Scaling)
scaler = StandardScaler()
df_standard = df.copy()
df_standard[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Printing Scaled Data
print("\nDataFrame after Min-Max Scaling:")
print(df_minmax.head())

print("\nDataFrame after Standardization:")
print(df_standard.head())

# Adult Income Dataset
df1 = pd.read_csv('adult.csv')

# Loading and Inspecting the Dataset
df1.head()
df1.shape
print(df1.info())
print(df1.describe())

# Missing Value Detection
missing_values = df1.isnull().sum()
print(missing_values[missing_values > 0])

# Detecting and Encoding Categorical Columns
categorical_cols = df1.select_dtypes(include=['object']).columns
print("Categorical columns identified:", categorical_cols)

if len(categorical_cols) > 0:
    df1 = pd.get_dummies(df1, columns=categorical_cols, drop_first=True)
    print("\nDataFrame after one-hot encoding:")
    print(df.head())
else:
    print("\nNo categorical columns found in the dataset.")

# Scaling Numerical Features
numerical_cols = df1.select_dtypes(include=['number']).columns

# Feature Scaling
numerical_cols = df.select_dtypes(include=['number']).columns

# a. Min-Max Scaling
scaler = MinMaxScaler()
df_minmax = df.copy()
df_minmax[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# b. Standardization (Z-score Scaling)
scaler = StandardScaler()
df_standard = df.copy()
df_standard[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Printing Scaled Data
print("\nDataFrame after Min-Max Scaling:")
print(df_minmax.head())

print("\nDataFrame after Standardization:")
print(df_standard.head())
