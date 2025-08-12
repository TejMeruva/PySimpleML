import pandas as pd
import numpy as np

def confusionMatrix(y, target):

def recallScore(y, target):
    pass

def accuracyScore(y, target):
    return (y == target).sum()/y.shape[0]

def precisionScore(y, target):
    pass

def f1Score(y, target):
    pass

def RSMEScore(y, target):
    y = y.to_numpy()
    target = target.to_numpy()
    return np.sqrt(((y - target) ** 2).mean())

def R2Score(y, target):
    y = y.to_numpy()
    target = target.to_numpy()
    SSRes = ((y - target) ** 2).sum()
    SSTot = ((target - target.mean()) ** 2).sum()
    return 1 - (SSRes / SSTot)