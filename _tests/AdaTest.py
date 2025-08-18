from PySimpleML.models.AdaBoost import Stump, AdaBoostModel
from PySimpleML.models.DT import _giniScore
from PySimpleML.scores import f1Score

import pandas as pd
import numpy as np

myStump = Stump(1)

data = pd.read_csv('/Users/msreeramulu/SWD/Python/PySimpleML/Examples/data/Flowers.csv')
dataShuff = data.sample(frac=1).reset_index(drop=True)
trainData = dataShuff.iloc[:140, :] 
testData = dataShuff.iloc[140:, :]
XTrain = trainData.iloc[:, 1:-1]
yTrain = trainData.iloc[:, [-1]]
XTest = testData.iloc[:, 1:-1]
yTest = testData.iloc[:, -1]

model = AdaBoostModel(50, 1)
model.train(XTrain, yTrain)
# weights = np.full_like(yTest, 1)
# weights[:3, :] = weights[:3, :] * 2
# weights = weights / weights.sum()
# # print(weights)
pred = model.predict(XTest)
print(f1Score(pred, yTest))

 
