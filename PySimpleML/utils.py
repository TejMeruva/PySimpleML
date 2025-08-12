import pandas as pd
import numpy as np

def normalizeDF(df: pd.DataFrame, mean=-1, std=1)-> pd.DataFrame:
    if not isinstance(mean, pd.Series): mean = df.mean()
    if not isinstance(std, pd.Series) : std = df.std()
    return (df - mean)/std, mean, std

def deNormalizeDF(df: pd.DataFrame, mean, std)-> pd.DataFrame:
    return (df * std) + mean

def normalizeNP(arr:np.ndarray, mean=np.array([0]), std=np.array([0])) -> np.ndarray:
    if not 0 in mean.shape: mean = arr.mean(axis=0)
    if not 0 in std.shape : std = arr.std(axis=0)
    return (arr-mean)/std, mean, std

def deNormalizeNP(arr:np.ndarray, mean, std) -> np.ndarray:
    return arr * std + mean

def euclidDist(a: np.ndarray, b:np.ndarray):
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    delta =  a-b
    deltaSq = (a-b)**2
    dist = (deltaSq.sum())**0.5
    return dist