import numpy as np

def l1_og(vector):
    s = 0
    for i in vector:
        s += ((i) ** 2) ** 0.5
    return s

def L1(v):
    if isinstance(v, (list, np.array)):
        raise TypeError('input must be list or np.array')
        
    return np.sum(np.fabs(v))

def l2_og(vector):
    s = 0
    for i in vector:
        s += (i ** 2)
    return s ** 0.5

def L2(v):
    if isinstance(v, (list, np.array)):
        raise TypeError('input must be list or np.array')

    return np.sqrt(np.sum(np.array(v) ** 2))

def L2Dot(v):
    if isinstance(v, (list, np.array)):
        raise TypeError('input must be list or np.array')
    s = np.dot(v, v)
    return np.sqrt(s)

def RMSE(h, y):
    if len(h) != len(y):
        raise ValueError('Vectors must be same length')

    return L2(h-y) * np.sqrt(1 / len(h))

def MAE(h, y):
    if len(h) != len(y):
        raise ValueError('Vectors must be same length')

    return 1 / len(h) * L1(h-y)