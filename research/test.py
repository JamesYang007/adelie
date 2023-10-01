import numpy as np
from dataclasses import dataclass


@dataclass
class A:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    
    def __init__(self, x):
        self.x = x
        self.y = x
        self.z = x

x = np.array([1,2,3])
a = A(x)
for attr in dir(a):
    if not callable(getattr(a, attr)):
        print(attr)