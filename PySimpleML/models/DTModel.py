import pandas as pd
import numpy as np 
from ..utils import isNum, valueCounts

class _Question:
    def __init__(self, serInd, value, cols):
        self.serInd = serInd
        self.value = value
        self.cols = cols

    def test(self, value:np.ndarray):
        # print(isNum(value)),
    
        isNumVect = np.vectorize(isNum, otypes=[bool])
        if isNumVect(value).sum() == value.size:
            return (value >= self.value)
        else:
            return (value == self.value)
    def __str__(self):
        if isNum(self.value):
            return f'is {self.cols[self.serInd]} >= {self.value}?'
        else:
            return f'is {self.cols[self.serInd]} == {self.value}?'
        
# def _giniScore(vals):
#     vals = list(vals)
#     unique = list(set(vals))
#     score = 1
#     for val in unique:
#         score -= (vals.count(val)/len(vals))**2
#     return score

def _giniScore(y: np.ndarray):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

def _split(data:np.ndarray, question:_Question):
    ser = data[:, [question.serInd]].reshape(-1)
    mask = question.test(ser)
    return data[mask, :], data[~mask, :]

def _splitInds(data:np.ndarray, question:_Question):
    ser = data[:, [question.serInd]].reshape(-1)
    mask = question.test(ser)
    return mask, ~mask

def _infoGain(data:np.ndarray, question:_Question):
    score = _giniScore(data[:, [-1]])
    trueData, falseData = _split(data, question)
    wmean = (trueData.shape[0] * _giniScore(trueData[:, -1]) + falseData.shape[0] * _giniScore(falseData[:, -1]))/(trueData.shape[0] + falseData.shape[0])
    return score - wmean

def _bestQuestion(X:np.ndarray, cols):
    bestInfo = -1
    bestQ = None
    for colInd in range(X.shape[1]):
        for val in np.unique(X[:, colInd]).reshape(-1):
            ques = _Question(colInd, val, cols)
            info = _infoGain(X, ques)
            if info > bestInfo:
                bestInfo  = info
                # print(bestInfo)
                bestQ = ques
                # print(bestQ)
    # print(bestInfo)
    return bestQ

class DecisionNode:
    def __init__(self, question, trueNext, falseNext):
        self.question = question
        self.trueNext = trueNext
        self.falseNext = falseNext

    def __str__(self):
        return str(self.question)
    
class Leaf:
    def __init__(self, y:np.ndarray, task):
        self.labels, self.counts= valueCounts(y)
        self.y = y
        self.task = task
    def __str__(self):

        match self.task:
            case 0:
                return self.y.mean()
            case 1:
                return self.labels[np.argmax(self.counts)]
            case _:
                pass

class DecisionTree:
    def __init__(self, task=0):
        self.task = task

    def train(self, X:pd.DataFrame, y:pd.DataFrame):
        tdata = pd.concat([X, y], axis=1).to_numpy()
        # print(tdata)
        # print(tdata)
        cols = list(X.columns)
        self.rootNode = self._buildTree(tdata, self.task, cols)

    def _buildTree(self, data:np.ndarray, task, cols):
        q = _bestQuestion(data[:, :-1], cols)
        info = _infoGain(data, q)
        if info == 0: return Leaf(data[:, -1], task)
        trueData, falseData = _split(data, q)
        # print(q)
        # print('gay')
        # print(trueData)
        trueBranch = self._buildTree(trueData, task, cols)
        falseBranch = self._buildTree(falseData, task, cols)
        return DecisionNode(q, trueBranch, falseBranch)
    
    def predict(self, inp:pd.DataFrame):
        inpNP = inp.to_numpy()
        inds = np.arange(0, inpNP.shape[0], 1).reshape(-1, 1)
        inpNPInded = np.hstack([inpNP, inds])
        op = self._op(inpNPInded, self.rootNode)
        return pd.Series(op[op[:, -1].argsort(), :-1].reshape(-1))
    def _op(self, inp: np.ndarray, node: DecisionNode):
        # if inds == None: inds = np.arange(0, inp.shape[0]).reshape(-1, 1)
        # inpInded = np.hstack([inds, inp])
        trueData, falseData = _split(inp, node.question)
        if isinstance(node.trueNext, Leaf) and isinstance(node.falseNext, Leaf):
            trueInds = trueData[:, [-1]]
            falseInds = falseData[:, [-1]]
            return np.vstack([
                np.hstack([np.full_like(trueInds, str(node.trueNext), dtype=object), trueInds]),
                np.hstack([np.full_like(falseInds, str(node.falseNext), dtype=object), falseInds])
            ])
        if isinstance(node.trueNext, Leaf) and not isinstance(node.falseNext, Leaf):
            trueInds = trueData[:, [-1]]
            falseInds = falseData[:, [-1]]
            return np.vstack([
                np.hstack([np.full_like(trueInds, str(node.trueNext), dtype=object), trueInds]),
                self._op(falseData, node.falseNext)
            ])
        if not isinstance(node.trueNext, Leaf) and isinstance(node.falseNext, Leaf):
            trueInds = trueData[:, [-1]]
            falseInds = falseData[:, [-1]]
            return np.vstack([
                self._op(trueData, node.trueNext),
                np.hstack([np.full_like(falseInds, str(node.falseNext), dtype=object), falseInds])
            ])
        if not isinstance(node.trueNext, Leaf) and not isinstance(node.falseNext, Leaf):
            trueInds = trueData[:, [-1]]
            falseInds = falseData[:, [-1]]
            return np.vstack([
                self._op(trueData, node.trueNext),
                self._op(falseData, node.falseNext)
            ])


    def _leafToNP(self, X):
        return

    def fn(self, node, ind=0):
        s = ''
        s += f'{str(node)}\n'
        if isinstance(node.trueNext, Leaf):
            s += (ind+1)*'\t' + f'(if True) {str(node.trueNext)}' + '\n'
        else:
            s+= (ind+1)*'\t'  + f'(if True) '
            s+= self.fn(node.trueNext, ind+1)
        if isinstance(node.falseNext, Leaf):
            s += (ind+1)*'\t' + f'(if False) {str(node.falseNext)}' + '\n'
        else:
            s+= (ind+1)*'\t' +  f'(if False) '
            s+= self.fn(node.falseNext, ind+1)
        return s
    def __str__(self):
        return self.fn(self.rootNode)




