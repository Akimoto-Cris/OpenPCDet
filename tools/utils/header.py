import sys
import torch
import importlib
import numpy as np

NaN = float('nan')
Inf = float('inf')


def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse
