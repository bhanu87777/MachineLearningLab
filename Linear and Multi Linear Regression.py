import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- Linear Regression (Single Feature) ---
df1 = pd.read_csv('Housing.csv')  # columns: area, price

X1 = df1[['area']]   # feature
y1 = df1['price']    # target

model1 = LinearRegression()
model1.fit(X1, y1)

# Predict for area = 3300 and 5000
pred_3300 = model1.predict(pd.DataFrame({'area':[3300]}))[0]
pred_5000 = model1.predict(pd.DataFrame({'area':[5000]}))[0]

# Plot for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(X1, y1, color='blue', marker='o', label='Data Points')
plt.plot(X1, model1.predict(X1), color='red', label='Best Fit Line')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.title('Linear Regression (Area vs Price)')
plt.legend()
plt.grid(True)
plt.savefig('linear_regression_plot.png')  # Save plot as image
plt.show()


# --- Multiple Linear Regression ---
df2 = pd.read_csv('Housing.csv')  # columns: area, bedrooms, age, price

# Fill missing values in bedrooms if any
df2['bedrooms'] = df2['bedrooms'].fillna(df2['bedrooms'].median())

X2 = df2[['area', 'bedrooms']]  # features
y2 = df2['price']

model2 = LinearRegression()
model2.fit(X2, y2)

# Predict for multiple features
multi_pred = model2.predict(pd.DataFrame({'area':[3000], 'bedrooms':[3]}))[0]

# Plot for Multiple Linear Regression: 
# Here we visualize predicted price for different bedroom counts with fixed area

bedroom_vals = np.arange(int(df2.bedrooms.min()), int(df2.bedrooms.max())+1)
pred_prices = model2.predict(pd.DataFrame({'area':[3000]*len(bedroom_vals), 'bedrooms':bedroom_vals}))

plt.figure(figsize=(8, 6))
plt.plot(bedroom_vals, pred_prices, marker='o', linestyle='-', color='green')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Predicted Price')
plt.title('Multiple Linear Regression\n(Predicted Price vs Bedrooms for area=3000 sq ft)')
plt.grid(True)
plt.savefig('multiple_linear_regression_plot.png')  # Save plot as image
plt.show()

# Print prediction results for clarity
print(f"Linear Regression Predictions:")
print(f"Price for 3300 sq ft: ₹{pred_3300:,.0f}")
print(f"Price for 5000 sq ft: ₹{pred_5000:,.0f}\n")

print(f"Multiple Linear Regression Prediction:")
print(f"Price for 3000 sq ft and 3 bedrooms: ₹{multi_pred:,.0f}")
