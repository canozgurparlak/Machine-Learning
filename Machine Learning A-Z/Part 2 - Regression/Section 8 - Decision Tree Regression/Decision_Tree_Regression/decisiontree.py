# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# print(dataset.head())
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# # Fitting the Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion='mse', random_state=0)
regressor.fit(X, y)

# # Predicting a new result
y_pred = regressor.predict([[6.5]])
print('Prediction with Decision Tree')
print(y_pred)

# # Visualising the Decision Tree Regression results
fig = plt.figure()
plt.figure(1)
plt.scatter(X, y, color='red', label='Real Salaries')
plt.plot(X, regressor.predict(X), color='blue', label='Predicted Salaries')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend()

# # Visualising the Decision Tree Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.figure(2)
plt.scatter(X, y, color='red', label='Real Salaries')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Predicted Salaries')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend()
plt.show()
