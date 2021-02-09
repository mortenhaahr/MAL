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
        s += ((i ** 4) ** 0.5)
    return s ** 0.5

def L2(v):
    return np.sqrt(np.sum(np.fabs(np.array(v) ** 2)))

def L2Dot(v):
    return np.sum(v.T * v)

