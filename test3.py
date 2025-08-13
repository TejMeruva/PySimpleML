from PySimpleML.models.DTModel import _Question, _giniScore, _split, _infoGain, _bestQuestion, DecisionTree
import pandas as pd
import numpy as np
from PySimpleML.models.DT import infoGain, Question, bestQuestion
from PySimpleML.scores import accuracyScore, f1Score

data = {
    "Feature1": [5.1, 4.9, 6.2, 5.9, 6.5, 5.0, 6.0, 5.5, 6.3, 5.8],
    "Feature2": [3.5, 3.0, 3.4, 3.2, 3.0, 3.6, 3.1, 3.5, 2.9, 3.3],
    "Feature3": [1.4, 1.4, 4.5, 4.2, 5.5, 1.4, 4.0, 1.3, 5.6, 4.4],
    "Class":    ["A", "A", "B", "B", "C", "A", "B", "A", "C", "B"]
}

data = pd.DataFrame(data)

# print(data)

q = _Question(3, 'A', data.columns)
# q2 = Question('Class', 'A') 
# print(bestQuestion(data.rename(columns={'Class':'Label'})))
# print(_bestQuestion(data.iloc[:, :-1].to_numpy(), list(data.columns)[:-1]))
X = data.iloc[:, :-1]
y = data.iloc[:, [-1]]
tree = DecisionTree(1)
tree.train(X, y)
# print(tree)
pred = tree.predict(X)
print(pd.concat([pred.reset_index(drop=True), y.reset_index(drop=True)], axis=1))