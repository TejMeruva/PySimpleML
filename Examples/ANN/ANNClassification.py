import pandas as pd
from PySimpleML.models.ANN import NeuralNetwork
from PySimpleML.scores import f1Score

data = pd.read_csv('/Users/msreeramulu/SWD/Python/PySimpleML/Examples/data/Flowers.csv')
dataShuff = data.sample(frac=1).reset_index(drop=True)
trainData = dataShuff.iloc[:145, :] 
testData = dataShuff.iloc[145:, :]
XTrain = trainData.iloc[:, :-1]
yTrain = trainData.iloc[:, [-1]]
XTest = testData.iloc[:, :-1]
yTest = testData.iloc[:, -1]

model = NeuralNetwork(10, 10, task=1)
model.train(XTrain, yTrain, 0.1, 100)
pred = model.predict(XTest, withConfidences=True)
print(pred.head())
print(f1Score(pred['Label'], yTest))
