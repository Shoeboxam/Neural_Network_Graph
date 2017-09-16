import numpy as np
from .Gate import Gate


class Variable(Gate, np.ndarray):
    """Datatype for differentiable variables"""
    # Custom operations for 3D and certain non-conformable arrays
    # Enforces mutability for all numerics
    parents = []

    def __new__(cls, a):
        obj = np.array(a).view(cls)
        return obj

    # This is a seed in the recursion of the graph network
    def __call__(self, *args):
        return self

    @property
    def output_nodes(self):
        return self.shape[0]
