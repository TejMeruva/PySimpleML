import numpy as np
import pandas as pd
from ..utils import normalizeNP, deNormalizeNP, euclidDist
import json

class KNNModel:
    def __init__(self, k=0):
        self.k = k

    def train(self, X: pd.DataFrame, y: pd.DataFrame):
        self.ycols = y.columns
        self.X, self.xmeans, self.xstds = normalizeNP(X.to_numpy())
        self.y, self.ymeans, self.ystds  = normalizeNP(y.to_numpy())

    def _predictOne(self, inp: np.ndarray):
        distFromX = lambda row: euclidDist(row, inp)
        dists = np.apply_along_axis(distFromX, axis=1, arr=self.X)
        nearInds = dists.argpartition(self.k)[:self.k]
        return deNormalizeNP(self.y[nearInds].mean(axis=0), self.ymeans, self.ystds)
    
    def predict(self, inp: pd.DataFrame):
        inpNorm = normalizeNP(inp.to_numpy(), self.xmeans, self.xstds)[0]
        preds = np.apply_along_axis(self._predictOne, axis=1, arr=inpNorm)
        return pd.DataFrame(preds, columns=[f'{x}Pred' for x in self.ycols])
                           
    def save(self, fileName: str):
        d = dict()
        d['k'] = self.k
        d['ycols'] = list(self.ycols)
        d['xmeans'] = self.xmeans.tolist()
        d['xstds'] = self.xstds.tolist()
        d['ymeans'] = self.ymeans.tolist()
        d['ystds'] = self.ystds.tolist()
        d['X'] = self.X.tolist()
        d['y'] = self.y.tolist()
        with open(fileName, 'w') as file:
            json.dump(d, file)

    def load(self, fileName: str):
        with open(fileName, 'r') as file:
            d = json.load(file)
        self.k = d['k']
        self.ycols = d['ycols']
        self.xmeans = np.array(d['xmeans'])
        self.xstds = np.array(d['xstds'])
        self.ymeans = np.array(d['ymeans'])
        self.ystds = np.array(d['ystds'])
        self.X = np.array(d['X'])
        self.y = np.array(d['y'])
