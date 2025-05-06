from deepdig.layers.activation import Activation
import math
import numpy as np

class Sigmoid(Activation):

    def compute(self, Z: np.ndarray) -> np.ndarray:

        """
        Passes Z through a sigmoid function and returns the values in an np.ndarray
        >>> relu = ReLU()
        >>> l = np.array([-12, 0, 1, 24])
        >>> return relu.compute(l))
        >>> [6.14417460e-06 5.00000000e-01 7.31058579e-01 1.00000000e+00]
        """

        sigmoid = 1/(1+np.exp(-Z)) # equivalent of [1/(1+e**(-x)) for x in Z]
        
        return sigmoid
    
    def derivative(self, z: np.ndarray) -> np.ndarray:
        """computes σ(z) = a and returns it's derivative as an np.ndarray"""
        a = compute(z)
        return s*(1-a)
