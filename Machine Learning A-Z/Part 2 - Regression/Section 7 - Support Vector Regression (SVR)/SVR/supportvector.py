# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values
# y = np.array([y])
# y = y.reshape(-1, 1)
# print(dataset.head())

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf', gamma='auto')
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
y_pred = sc_y.inverse_transform(y_pred)
print('Predicting with SVR')
print(y_pred)

# # Visualising the SVR results
# fig = plt.figure()
# plt.figure(1)
# plt.scatter(X, y, color='red', label='Real Salaries')
# plt.plot(X, regressor.predict(X), color='blue', label='Predicted Salaries')
# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.legend()


# # Visualising the SVR results (for higher resolution and smoother curve)
# plt.figure(2)
# X_grid = np.arange(min(X), max(X), 0.01)  # choice of 0.01 instead of 0.1 step because the data is feature scaled
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color='red', label='Real Salaries')
# plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Predicted Salaries')
# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.legend()
# plt.show()

# Visualising the SVR results
X = sc_X.inverse_transform(X)
y = sc_y.inverse_transform(y)
fig = plt.figure()
plt.figure(1)
plt.scatter(X, y, color='red', label='Real Salaries')
plt.plot(X, sc_y.inverse_transform(regressor.predict(sc_X.transform(X))), color='blue', label='Predicted Salaries')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend()


# Visualising the SVR results (for higher resolution and smoother curve)
plt.figure(2)
X_grid = np.arange(min(X), max(X), 0.01)  # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red', label='Real Salaries')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color='blue', label='Predicted Salaries')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend()
plt.show()
