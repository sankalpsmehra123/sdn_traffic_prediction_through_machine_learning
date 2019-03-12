# Platform: `linux`
# Random Forest Regression
# Importing the libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(
    '/home/shadowziyus/Documents/'
    'SDN-Traffic-Prediction-Through-Machine-Learning/Dataset.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(x_train, y_train)

# predictions
y_pred = regressor.predict(x_test)

# Visualising the Random Forest Regression results (higher resolution)
x_grid = np.arange(min(x_train), max(x_train), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x_train, y_train, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Traffic Prediction (Random Forest)')
plt.xlabel('Traffic')
plt.ylabel('Time')
plt.show()
