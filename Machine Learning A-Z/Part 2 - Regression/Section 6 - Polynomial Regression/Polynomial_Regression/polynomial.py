# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')
# plt.style.use('fivethirtyeight')

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# print(dataset.head())
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
# make sure that X is a matrix and y is a vector

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dateset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualising the Linear Regression results
fig = plt.figure()
plt.figure(1)
plt.scatter(X, y, color='red', label='Real Salaries')
plt.plot(X, lin_reg.predict(X), color='blue', label='Predicted Salaries')
plt.title('Truth or Bluff  (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()

# Visualisng the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.figure(2)
plt.scatter(X, y, color='red', label='Real Salaries')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue', label='Predicted Salaries')
plt.title('Truth or Bluff  (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
# plt.show()

# Predicting a new result with Linear Regression
print('Predicting with Linear Regression')
print(lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print('Predicting with Polynomial Regression')
print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))

# Coefficients
print('Coefficients for Linear Regression')
print(lin_reg.coef_)
print('Coefficients for Polynomial Regression')
print(lin_reg2.coef_)
