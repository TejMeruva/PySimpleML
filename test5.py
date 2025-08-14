from PySimpleML.models.RF import _bootstrap, _randBestQuestion, RandomForest
from PySimpleML.models.DT import DecisionTree
from PySimpleML.scores import f1Score
import pandas as pd
import numpy as np 

data = pd.read_excel('/Users/msreeramulu/SWD/Python/PySimpleML/ClassData.xlsx').drop(columns=['Id'])
X = data.iloc[:, :-1]
y = data.iloc[:, [-1]]
dataNP = data.to_numpy()
rf = RandomForest(100, 3, 1)
node = rf._buildTree(_bootstrap(dataNP), np.array(data.columns))
# print(data.shape)
rf.train(X, y)
pred = rf.predict(X)
print(f1Score(pred, y))

from PySimpleML.models.ANN import NeuralNetwork
print()
nn = NeuralNetwork(10, 10, task=1)
nn.train(X, y, 0.1, 10)
pred = nn.predict(X)
print(f1Score(pred, y))