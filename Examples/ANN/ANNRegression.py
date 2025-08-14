from PySimpleML.models.ANN import NeuralNetwork
from PySimpleML.scores import R2Score
import pandas as pd

data = pd.read_csv('/Users/msreeramulu/SWD/Python/PySimpleML/Examples/data/Houses.csv')
dataShuff = data.sample(frac=1)
trainData = data.iloc[:20000, :] #using only 10k rows out of~20000
testData = data.iloc[20000:, :]
XTrain = trainData.iloc[:, 1:]
yTrain = trainData.iloc[:, [0]]
XTest = testData.iloc[:, 1:]
yTest = testData.iloc[:, [0]]

model = NeuralNetwork(10, 10, task=0)
model.train(XTrain, yTrain, 0.01, 5)
pred = model.predict(XTest)
print(R2Score(pred, yTest))
