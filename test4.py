
from PySimpleML.scores import f1Score, precisionScore, accuracyScore
import pandas as pd
from PySimpleML.models.DT import DecisionTree as DT1
from PySimpleML.models.DTModel import DecisionTree as DT2

data = pd.read_excel('/Users/msreeramulu/SWD/Python/PySimpleML/ClassData.xlsx')
dataShuff = data.rename(columns={'Species':'Label'}).reset_index(drop=True)
# dataShuff.drop(['Id'], inplace=True)
trainData = dataShuff.iloc[:140, :]
testData = dataShuff.iloc[140:, :]
# print(data.info())
Xtrain = trainData.iloc[:, 1:-1]
ytrain = trainData.iloc[:, [-1]]
Xtest = testData.iloc[:, 1:-1]
ytest = testData.iloc[:, [-1]]
# print(Xtest.head())

#DT1
# tree = DT1(pd.concat([Xtrain, ytrain], axis=1), 1)
# # tree.train(Xtrain, ytrain)
# pred = tree.predict(Xtest)
# print(f1Score(pd.Series(pred), ytest))
# print(tree)

#DT2 (10 times faster exp)
tree = DT2(1)
tree.train(Xtrain, ytrain)
pred = tree.predict(Xtest)
print(f1Score(pred, ytest))
print(tree)