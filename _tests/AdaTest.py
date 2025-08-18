from PySimpleML.models.AdaBoost import Stump, AdaBoostModel

import pandas as pd

myStump = Stump(1)

data = pd.read_csv('/Users/msreeramulu/SWD/Python/PySimpleML/Examples/data/Flowers.csv')
dataShuff = data.sample(frac=1).reset_index(drop=True)
trainData = dataShuff.iloc[:145, :] 
testData = dataShuff.iloc[145:, :]
XTrain = trainData.iloc[:, 1:-1]
yTrain = trainData.iloc[:, [-1]]
XTest = testData.iloc[:, 1:-1]
yTest = testData.iloc[:, -1]

model = AdaBoostModel(1, 1)
model.train(XTrain, yTrain)
print(model.stumps[0])
