
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("boston_train.csv")
dataset
dataset.head()

dataset.info()
# dataset.describe()

dataset = dataset.drop('ID',axis=1)

dataset.plot.scatter('rm', 'medv')
# dataset.plot.scatter('dis', 'medv')

sns.heatmap(dataset.corr(), cmap = 'RdGy')

#split x and y
x=dataset[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'black', 'lstat']]
y = dataset['medv']

# split train and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

X_train.head()

lr = LinearRegression()
lr.fit(X_train,y_train)
# print(lr)

predictions = lr.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

x=dataset[['crim', 'indus', 'rm', 'age',  'tax','ptratio', 'lstat']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train,y_train)

predictions = lr.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))