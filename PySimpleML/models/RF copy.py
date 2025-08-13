from DT import DecisionTree, infoGain, Question, bestQuestion, Leaf, split, DecisionNode
import pandas as pd
import random
import numpy as np

def bootstrap(data):
    return data.sample(data.shape[0], replace=True).reset_index(drop=True)

def randBestQuestion(data, nvar: int, ignore: list):
    cols = list(data.columns)[:-1]
    for elem in ignore: 
        cols.remove(elem)
    random.shuffle(cols)
    colSet = cols[:nvar]
    return bestQuestion(data[colSet + ['Label']])

class RandomForrest:
    def __init__(self, ntrees, nvar, mode):
        self.ntrees = ntrees
        self.nvar = nvar
        self.mode = mode

    def growTree(self, data, toIgnore=[]):
        q = randBestQuestion(data, self.nvar, toIgnore)
        if q == -1: return Leaf(data, self.mode)
        info = infoGain(data, q)
        if info == 0: return Leaf(data, self.mode)
        ig = list(toIgnore)
        ig.append(q.column)
        trueData, falseData = split(data, q)
        trueBranch = self.growTree(trueData, ig)
        falseBranch = self.growTree(falseData, ig)
        return DecisionNode(q, trueBranch, falseBranch)
    
    def growForrest(self, data):
        trees = []
        print('Started!')
        for i in range(self.ntrees):
            bdata = bootstrap(data)
            tree = self.growTree(bdata)
            trees.append(DecisionTree(rootNode=tree))
            print(i+1, end='\r')
        self.trees = trees

    def predict(self, inp):
        rows = []
        for tree in self.trees:
            rows.append(tree.predict(inp))
        
        match self.mode:
            case 0:
                rows = pd.DataFrame(np.array(rows).T.astype('float'))
                return rows.mean(axis=1)
            case 1:
                rows = pd.DataFrame(np.array(rows).T)
                return rows.mode(axis=1)
            
     
        