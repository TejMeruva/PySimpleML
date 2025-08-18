from .DT import DecisionTree, _bestQuestion, DecisionNode, _split, _infoGain, Leaf
import numpy as np
import pandas as pd
from .BaseModel import MLModel

class Stump(DecisionTree):
    def __init__(self, task, rootNode=None):
        super().__init__(1, rootNode)
        self.say = 0

    def _train(self, X, y, cols, weights):
        data = np.hstack([X, y])
        q = _bestQuestion(data, cols, weights)
        # print(_infoGain(data, q, weights))
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
        self.labels = np.unique(y)
        X = X.to_numpy()
        y = y.to_numpy()
        weights = np.full_like(y, 1.0, dtype=np.float64)
        weights = weights/weights.sum()
        allWeights = weights.copy()
        for i in range(self.stumpCount):
            stump = Stump(self.task)
            stump.train(X, y, cols, weights)
            self.stumps.append(stump)
            pred = stump.predictNP(X)
            totalError = weights[~(y == pred)].sum()
            say = 0.5 * np.log((1-totalError)/(totalError))
            stump.say = say
            
            weights[y==pred] = weights[y==pred] * (np.e ** (-stump.say))
            weights[~(y==pred)] = weights[~(y==pred)] * (np.e ** stump.say)
            weights = weights / weights.sum()
            allWeights = np.hstack([allWeights, weights])
            # print(np.hstack([(y==pred), weights]))
        # print(allWeights)
        


    def _predict(self, X:pd.DataFrame, *args, **kwargs):
        X = X.to_numpy()
        preds = self.stumps[0].predictNP(X)
        labels = self.labels
        sigSay = [0 for label in labels]
        says = np.array([self.stumps[0].say])
        
        for stump in self.stumps[1:]:
            pred = stump.predictNP(X)
            says = np.append(says, stump.say)
            preds = np.hstack([preds, pred])
        labels  = np.unique(preds)
        ops = []
        def toClass(row) -> str:
            for ind in range(len(labels)):
                sigSay[ind] = says[(row == labels[ind])].sum()
            ops.append(labels[np.array(sigSay).argmax()])
            return 0
        
        np.apply_along_axis(toClass, axis=1, arr=preds)

        return pd.Series(ops)
    
    def __str__(self):
        s = ''
        for stump in self.stumps:
            s += str(stump) + f' (say: {stump.say})' + '\n' * 2
        return s

    