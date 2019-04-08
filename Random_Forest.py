# Platform: `linux`
# Random Forest Regression
# Importing the libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(
    '/home/shadowziyus/Documents/'
    'SDN-Traffic-Prediction-Through-Machine-Learning/Dataset.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 4].values

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
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(x_train, y_train)

# predictions
y_pred = regressor.predict(x_test)

# seprating
sec = x_test[:, 0]

# Visualizing Random Forest for the dataset
plt.scatter(sec, y_pred, color='blue')
plt.savefig(
    '/home/shadowziyus/Documents/'
    'SDN-Traffic-Prediction-Through-Machine-Learning/predicted.png')
plt.show()
plt.scatter(sec, y_test, color='red')
plt.savefig(
    '/home/shadowziyus/Documents/'
    'SDN-Traffic-Prediction-Through-Machine-Learning/actual.png')
plt.show()

error = y_pred - y_test
plt.scatter(sec, error, color='black')
plt.savefig(
    '/home/shadowziyus/Documents/'
    'SDN-Traffic-Prediction-Through-Machine-Learning/error.png')
plt.show()
