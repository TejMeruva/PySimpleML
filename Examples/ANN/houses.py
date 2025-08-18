import pandas as pd
from PySimpleML.models.DT import DecisionTree
from PySimpleML.scores import R2Score

data = pd.read_csv('/Users/msreeramulu/SWD/Python/PySimpleML/Examples/data/Houses.csv')
dataShuff = data.sample(frac=1)
trainData = data.iloc[:200, :] #using only 10k rows out of~20000
testData = data.iloc[20000:, :]
#training
XTrain = trainData.iloc[:, 1:]
yTrain = trainData.iloc[:, [0]]
#testing
XTest = testData.iloc[:, 1:]
yTest = testData.iloc[:, [0]]

model = DecisionTree(task=0)
model.train(XTrain, yTrain)
pred = model.predict(XTest)

print(R2Score(pred, yTest))
print(pred.head())
print(yTest.head())

