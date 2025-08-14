import pandas as pd
import numpy as np

def _mapping(y: pd.Series, target: pd.Series):
    if not isinstance(target, pd.Series) or not isinstance(y, pd.Series): raise ValueError('`target` and `y` arguments should be a Pandas Series')
    unique = np.unique(np.concatenate([y.unique(), target.unique()]))
    mapping = {v:k for k, v in dict(enumerate(unique)).items()}
    return mapping

def confusionMatrix(y: pd.Series, target: pd.Series):
    ynp = y.to_numpy().reshape(-1, 1)
    targetnp = target.to_numpy().reshape(-1, 1)
    ytar = np.hstack([ynp, targetnp])
    mapping = _mapping(y, target)
    cm = np.zeros((len(mapping), len(mapping)), dtype=int)
    def mapThis(x):
        return mapping[x]
    mapThis = np.vectorize(mapThis)
    ytar = mapThis(ytar)
    def update(col):
        cm[col[0], col[1]] += 1
        return 0
    np.apply_along_axis(update, axis=1, arr=ytar)
    return cm

def _TP(y, target, tarLabel):
    mapping = _mapping(y, target)
    ind = mapping[tarLabel]
    cm = confusionMatrix(y, target)
    return cm[ind, ind]

def _TN(y, target, tarLabel):
    mapping = _mapping(y, target)
    ind = mapping[tarLabel]
    cm = confusionMatrix(y, target)
    return cm[:ind, :ind].sum() + cm[ind+1:, ind+1:].sum() + cm[ind+1:, :ind].sum() + cm[:ind, ind+1:].sum()

def _FP(y, target, tarLabel):
    mapping = _mapping(y, target)
    ind = mapping[tarLabel]
    cm = confusionMatrix(y, target)
    return cm[:ind, ind].sum() + cm[ind+1:, ind].sum()

def _FN(y, target, tarLabel):
    mapping = _mapping(y, target)
    ind = mapping[tarLabel]
    cm = confusionMatrix(y, target)
    return cm[ind, :ind].sum() + cm[ind, ind+1:].sum()

def _evalCM(y, target, tarLabel, cm):
    mapping = _mapping(y, target)
    ind = mapping[tarLabel]
    return {
        "TP": cm[ind, ind],
        "TN": cm[:ind, :ind].sum() + cm[ind+1:, ind+1:].sum() + cm[ind+1:, :ind].sum() + cm[:ind, ind+1:].sum(),
        "FP": cm[:ind, ind].sum() + cm[ind+1:, ind].sum(), 
        "FN": cm[ind, :ind].sum() + cm[ind, ind+1:].sum()
    }
def recallScore(y, target):
    #tp/(tp + fn)
    op = dict()
    mapping = _mapping(y, target)
    for tarLabel in mapping:
        evaln = _evalCM(y, target, tarLabel, confusionMatrix(y, target))
        tp, fn = evaln["TP"], evaln["FN"]
        if tp + fn == 0:
            op[tarLabel] = 0
        else:
            op[tarLabel] = tp / (tp + fn)
    return pd.Series(op)

def accuracyScore(y, target):
    y = y.to_numpy().reshape(-1)
    target = target.to_numpy().reshape(-1)
    return (y == target).sum()/y.shape[0]

def precisionScore(y, target):
    op = dict()
    mapping = _mapping(y, target)
    for tarLabel in mapping:
        evaln = _evalCM(y, target, tarLabel, confusionMatrix(y, target))
        tp, fp = evaln["TP"], evaln["FP"]
        if tp + fp == 0:
            op[tarLabel] = 0
        else:
            op[tarLabel] = tp / (tp + fp)
    return pd.Series(op)

def f1Score(y, target):
    prec = precisionScore(y, target)
    rec = recallScore(y, target)
    denom = prec + rec
    f1 = 2 * (prec * rec) / denom
    f1[denom == 0] = 0.0  # Avoid NaN when both are zero
    return f1 

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