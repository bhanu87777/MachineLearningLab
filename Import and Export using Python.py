import pandas as pd
# Import
iris_df = pd.read_csv('Iris.csv')
print("First 5 rows of the Iris dataset:")
print(iris_df.head())

# Export 
output_path = "iris_exported.csv"
iris_df.to_csv(output_path, index=False)
print(f"\nIris dataset has been exported successfully to '{output_path}'")

df = pd.read_csv('iris_exported.csv')
df.head()
