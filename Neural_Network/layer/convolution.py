from config import NP_BACKEND
from Neural_Network.node import Node

if NP_BACKEND == 'NUMPY':
    import numpy as np
elif NP_BACKEND == 'JAX':
    import jax.numpy as np


class Convolve(Node):
    def __init__(self, children, kernel):
        super().__init__(children)
        self.kernel = kernel

    def propagate(self, features):
        # Combine features from all children
        features = np.vstack(features)

        # Determine edge of kernel
        kernel_edge = np.array(self.kernel.shape) - 1
        if len(features.shape) == 3:
            kernel_edge = np.append(kernel_edge, np.array([0]))

        # Convert 3D image into 5D sampler [kernel elements, channel, sample index]
        shape = [*self.kernel.shape, *(features.shape - kernel_edge)]
        memory_offset = [*features.strides[:2], *features.strides]
        samples = np.lib.stride_tricks.as_strided(features, shape=shape, strides=memory_offset)

        # Contract over first two dimensions
        return np.einsum("ij...,ij->...", samples, self.kernel)

    def backpropagate(self, features, variable, grad):
        # Not implemented
        return grad
