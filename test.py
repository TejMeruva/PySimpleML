from PySimpleML.models.KNN import KNNModel
import pandas as pd
from PySimpleML.utils import euclidDist, normalizeNP, normalizeDF
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PySimpleML.scores import RSMEScore, R2Score
trainData = pd.read_csv('PySimpleML/california_housing_train.csv')

# X = trainData.iloc[:, :-1]
# y = trainData.iloc[:, [-1, -2]]
# model = KNNModel(5)
# model.train(X, y)
# x = X.iloc[[10], :]

#Sine Data
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.01, 100)
X = pd.DataFrame(x, columns=['x'])
y = pd.DataFrame(y, columns=['y'])
plt.scatter(X, y, label='Actual',marker='x', alpha=0.5)
model = KNNModel(5)
model.train(X, y)

# model.save('knnModel.json')
# model.load('knnModel.json')
pred = model.predict(X)
print(R2Score(pred, y))
plt.plot(X, pred, label='Predicted')
# plt.scatter(X, y, label='Actual', alpha=0.5)
plt.legend()
plt.show()
# distFromX = lambda row: euclidDist(row, x)
# dists = np.apply_along_axis(distFromX, axis=1, arr=X)
# df = pd.DataFrame(X)
# df = pd.concat([df, pd.Series(dists)], axis=1)
# print(df)
# pred = model.predict(X.head())
# actual = y.head()
# print(pd.concat([pred, actual], axis=1).corr())