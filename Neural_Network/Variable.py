import numpy as np
from .Gate import Gate


class Variable(Gate, np.ndarray):
    """Datatype for differentiable variables"""
    # Custom operations for 3D and certain non-conformable arrays
    # Enforces mutability for all numerics
    # Provides seeds for recursive calls in the graph network

    def __new__(cls, a):
        print(a)
        objList = [sup.__new__(typ) for sup in Variable.__bases__]
        for obj in objList[1:]:
            objList[0].__dict__.update(copy.deepcopy(obj.__dict__))
        objList[0].attr3 = 333
        return objList[0]
        super(np.ndarray, cls,)
        return super(Gate, cls).__new__(cls).view(cls)
        return np.array(a).view(cls)

    def __init__(self):
        super().__init__(children=[])
        print(self.parents)

    def __call__(self, stimulus, parent=None):
        # Split a variable with respect to multiple parents
        if np.isscalar(self):
            return self

        if parent in self.parents:
            cursor = 0

            for par in self.parents:
                if parent is par:
                    return self[cursor:cursor + parent.input_nodes]
                cursor += par.input_nodes
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
