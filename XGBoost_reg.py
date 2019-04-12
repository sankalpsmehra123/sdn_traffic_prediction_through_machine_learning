# Platform: `linux(Ubuntu 18.04)'
# python version: 3.6
# predicting SDN traffic using XGBoost
import xgboost as xgb
from sklearn.model_selection import train_test_split
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

# Fitting XGBoost Regression to the dataset
xgbr = xgb.XGBRegressor()
regressor = xgbr.fit(x_train, y_train)

# predictions
y_pred = regressor.predict(x_test)

# seprating seconds form the Dataset
sec = x_test[:, 0]

# Visualizing XGBoost for the dataset
plt.scatter(sec, y_pred, color='blue')
plt.savefig(
    '/home/shadowziyus/Documents/'
    'SDN-Traffic-Prediction-Through-Machine-Learning/XGB_predicted.png')
plt.show()
plt.scatter(sec, y_test, color='red')
plt.savefig(
    '/home/shadowziyus/Documents/'
    'SDN-Traffic-Prediction-Through-Machine-Learning/XGB_actual.png')
plt.show()

error = y_pred - y_test
plt.scatter(sec, error, color='black')
plt.savefig(
    '/home/shadowziyus/Documents/'
    'SDN-Traffic-Prediction-Through-Machine-Learning/XGB_error.png')
plt.show()
