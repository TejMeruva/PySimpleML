import pandas as pd

def isNum(x) -> bool:
    return isinstance(x, int) or isinstance(x, float)

class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value
    def test(self, inp):
        if isNum(inp):
            return inp >= self.value
        else:
            return inp == self.value
    def __str__(self):
        if isNum(self.value):
            return f'is {self.column} >= {self.value}?'
        else:
            return f'is {self.column} == {self.value}?'

def giniScore(vals):
    vals = list(vals)
    unique = list(set(vals))
    score = 1
    for val in unique:
        score -= (vals.count(val)/len(vals))**2
    return score

def split(data, q: Question):
    trueInds = []
    falseInds = []
    for ind, row in data.iterrows():
        # try:
        if q.test(row[q.column]):
            trueInds.append(ind)
        else:
            falseInds.append(ind)
        # except:
            
    trueData = data.loc[trueInds, :].reset_index(drop=True)
    falseData = data.loc[falseInds, :].reset_index(drop=True)
    return trueData, falseData

def infoGain(data, q):
    trueData, falseData = split(data, q)
    wmean = (trueData.shape[0] * giniScore(trueData['Label'])+ falseData.shape[0] * giniScore(falseData['Label']))/data.shape[0]
    return giniScore(data['Label']) - wmean

def bestQuestion(data):
    infos = []
    qs = []
    columns = list(data.columns)
    columns.remove('Label')
    if len(columns) == 0: return -1
    for col in columns:
        for ind in data.index:
            q = Question(col, data.loc[ind, col])
            infos.append(infoGain(data, q))
            qs.append(q)
    ind = infos.index(max(infos))
    # print(max(infos))
    return qs[ind]

class DecisionNode:
    def __init__(self, question, trueNext, falseNext):
        self.question = question
        self.trueNext = trueNext
        self.falseNext = falseNext

    def __str__(self):
        return str(self.question)

class Leaf:
    def __init__(self, data, mode):
        self.freq = data['Label'].value_counts().reset_index()
        self.data = data
        self.mode = mode
    def __str__(self):

        match self.mode:
            case 0:
                return str(self.data['Label'].mean())
            case 1:
                return self.freq.loc[0, 'Label']
            case _:
                pass
        

class DecisionTree:
    def __init__(self, data=pd.DataFrame([]), mode=0, rootNode=None):
        if rootNode == None:
            self.rootNode = self.buildTree(data, mode)
        else:
            self.rootNode = rootNode
        self.mode = mode

    def buildTree(self, data, mode):
        q = bestQuestion(data)
        info = infoGain(data, q)
        if info == 0: return Leaf(data, mode)
        trueData, falseData = split(data, q)
        trueBranch = self.buildTree(trueData, mode)
        falseBranch = self.buildTree(falseData, mode)
        return DecisionNode(q, trueBranch, falseBranch)
    
    def predict(self, inp):
        op = []
        for ind, row in inp.iterrows():
            node = self.rootNode
            while not isinstance(node, Leaf):
                if node.question.test(row[node.question.column]):
                    node = node.trueNext
                else:
                    node = node.falseNext
            op.append(str(node))
        return op

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