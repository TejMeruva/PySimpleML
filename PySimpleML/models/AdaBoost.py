from .DT import DecisionTree, _bestQuestion, DecisionNode, _split, _infoGain, Leaf
import numpy as np
from .BaseModel import MLModel

def newDataset(X:np.ndarray, y:np.ndarray, weights:np.ndarray) -> tuple:
    pass

class Stump(DecisionTree):
    def __init__(self, task, rootNode=None):
        super().__init__(1, rootNode)
        self.say = 0

    def _train(self, X, y, cols):
        data = np.hstack([X, y])
        q = _bestQuestion(data, cols)
        trueData, falseData = _split(data, q)
        trueLeaf = Leaf(trueData[:, -1], self.task)
        falseLeaf = Leaf(falseData[:, -1], self.task)
        rootNode = DecisionNode(q, trueLeaf, falseLeaf)
        self.rootNode = rootNode


class AdaBoostModel(MLModel):
    def __init__(self, stumpCount=50, task=0):
        self.stumpCount = stumpCount
        self.task = task

    def _train(self, X, y, *args, **kwargs):
        self.stumps = []
        cols = X.columns
        X = X.to_numpy()
        y = y.to_numpy()
        for i in range(self.stumpCount):
            stump = Stump(self.task)
            stump.train(X, y, cols)
            pred = stump.predict(X)
            self.stumps.append(stump)
            pred = stump.predictNP(X)
            errorCount = (~(y == pred)).sum()
            weight = 1/y.shape[0]
            totalError = errorCount * weight
            say = 0.5 * np.log((1-totalError)/(totalError))
            stump.say = say
            weights = np.full_like(y, weight)
            weights[y==pred] = weight * (np.e ** (-say))
            weights[~(y==pred)] = weight * (np.e ** say)
            weights = weights / weights.sum()
            print(weights)


    def _predict(self, X, *args, **kwargs):
        return super()._predict(X, *args, **kwargs)
    