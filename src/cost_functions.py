import numpy as np
from activation_functions import Sigmoid

class QuadraticCost(object):
    def fn(a, y):
        return 0.5 * np.linalg.norm(a-y)**2

    def delta(z, a, y):
        return (a - y) * Sigmoid.delta(z)

class CrossEntropyCost(object):
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1-y) * np.log(1 - a)))

    def delta(z, a, y):
        return a - y
