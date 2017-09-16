import numpy as np
from .Gate import Gate


class Variable(Gate, np.ndarray):
    """Datatype for differentiable variables"""
    # Custom operations for 3D and certain non-conformable arrays
    # Enforces mutability for all numerics
    # Provides seeds for recursive calls in the graph network

    def __new__(cls, a):
        obj = np.array(a).view(cls)
        return obj

    def __call__(self, *args):
        return self

    def gradient(self, stimulus, variable, grad):
        if variable is not self:
            raise ValueError("The gradient should not have been backpropagated here. Not optimal.")
            # return grad * 0

        return grad  # @ np.eye(self.shape[0])

    @property
    def output_nodes(self):
        return self.shape[0]

    @property
    def variables(self):
        """List the input variables"""
        return []
