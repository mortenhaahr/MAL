import numpy as np

def l1_og(vector):
    s = 0
    for i in vector:
        s += ((i) ** 2) ** 0.5
    return s

def L1(v):
    return np.sum(np.fabs(v))

def l2_og(vector):
    s = 0
    for i in vector:
        s += (i ** 2)
    return s ** 0.5

def L2(v):
    return np.sqrt(np.sum(np.array(v) ** 2))

def L2Dot(v):
    s = np.dot(v, v)
    return np.sqrt(s)

def RMSE(h, y):
    assert len(h) == len(y), "Must be same length"
    return L2(h-y)

def MAE(h, y):
    assert len(h) == len(y), "Must be same length"
    return L1(h-y)