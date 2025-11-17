import random
import numpy as np

def get_fn(x, y):
    '''x=>0; y=>1'''
    # a * x + b = 0
    # a * y + b = 1
    # a (y - x) = 1
    a = 1 / (y - x)
    b = -x / (y - x)
    return lambda q: a * q + b

def parse(data):
    res = []
    mnx = 1e9 + 7
    mny = 1e9 + 7
    mnz = 1e9 + 7
    mxx = -mnx
    mxy = -mny
    mxz = -mnz

    for row in data:
        x, y, z = map(np.float32, row.split(','))
        mnx = min(mnx, x)
        mny = min(mny, y)
        mnz = min(mnz, z)
        mxx = max(mxx, x)
        mxy = max(mxy, y)
        mxz = max(mxz, z)

    fx = get_fn(mnx, mxx)
    fy = get_fn(mny, mxy)
    fz = get_fn(mnz, mxz)
    for row in data:
        x, y, z = map(np.float32, row.split(','))
        inp = np.array([np.array([fx(x)], dtype=np.float32), np.array([fy(y)], dtype=np.float32)], dtype=np.float32)
        out = np.array([np.array([fz(z)], dtype=np.float32)], dtype=np.float32)
        res.append((inp, out))

    return res

def load_data_wrapper(fname):
    with open(fname) as f:
        data = f.read().strip().split('\n')

    data = parse(data)

    sz = len(data) // 7
    random.shuffle(data)
    test = data[:sz]
    valid = data[sz:2*sz]
    train = data[2*sz:]

    return train, valid, test


