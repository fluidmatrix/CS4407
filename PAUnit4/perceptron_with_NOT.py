import numpy as np
from utils import step

def NOT(x1):
    weights = -1
    bias = 1
    output = np.dot([1], weights) + bias
    
    return step(output)

print('NOT Gate')
for x in [0,1]:
    print(x, '->', NOT(x))