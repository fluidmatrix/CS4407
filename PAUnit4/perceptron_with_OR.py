import numpy as np
from utils import step

def OR(x1, x2):
    weights = np.array([1,1])
    bias = -0.5
    output = np.dot([x1, x2], weights) + bias
    
    return step(output)

print("OR Gate")
for x1 in [0,1]:
    for x2 in [0,1]:
        print(x1, x2, '->', OR(x1,x2))