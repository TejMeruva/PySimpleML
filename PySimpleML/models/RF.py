import numpy as np
import pandas as pd
from PySimpleML.models.DT import _Question, _bestQuestion, _infoGain, Leaf, _split, DecisionNode, DecisionTree
from ..utils import valueCounts

def _bootstrap(data:np.ndarray):
    inds = np.random.randint(0, data.shape[0], size=data.shape[0])
    return data[inds, :]

def _randBestQuestion(data:np.ndarray, nvar: int, cols: np.ndarray, indsIncl:np.ndarray=None) -> _Question:
    if indsIncl is None : indsIncl = np.arange(0, data.shape[1], 1)
    inds = indsIncl[:-1]
    lessOrEqual = lambda x: x if x <= inds.size else inds.size
    inds = np.sort(np.random.choice(inds, lessOrEqual(nvar), replace=False))
    subData = data[:, np.append(inds, data.shape[1]-1).astype(int)]
    return _bestQuestion(subData, cols[inds.astype(int)])
class RandomForest:
    def __init__(self, ntrees, nvar, task):
        self.ntrees = ntrees
        self.nvar = nvar
        self.task = task
    def _buildTree(self, data:np.ndarray, cols: np.ndarray, inds=None):
        if inds is None: inds = np.arange(0, data.shape[1], 1)
        q = _randBestQuestion(data, self.nvar, cols, inds)
        if q is None: return Leaf(data[:, [-1]], self.task)
        info = _infoGain(data, q)
        if round(info, 10) == 0: return Leaf(data[:, [-1]], self.task)
        inds = inds[~(inds == q.serInd)]
        # print(inds)
        trueData, falseData = _split(data, q)
        trueBranch = self._buildTree(trueData,  cols, inds)
        falseBranch = self._buildTree(falseData, cols, inds)
        return DecisionNode(q, trueBranch, falseBranch)
    
    def train(self, X:pd.DataFrame, y:pd.DataFrame):
        cols = np.array(X.columns)
        X = X.to_numpy()
        y = y.to_numpy()
        tdata = np.hstack([X, y])
        trees = []
        for i in range(self.ntrees):
            bdata = _bootstrap(tdata)
            tree = DecisionTree(rootNode=self._buildTree(bdata, cols))
            trees.append(tree) 
        self.trees = trees

    def predict(self, X:pd.DataFrame):
        
        ops = pd.concat([tree.predict(X) for tree in self.trees], axis=1)
        match self.task:
            case 0:
                return None
            case 1:
                return(ops.mode(axis=1)[0])


        