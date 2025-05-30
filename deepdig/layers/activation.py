import numpy as np
import math
from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def compute(self, z):
        pass
    
    @abstractmethod
    def derivative(self, z):
        pass

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
        a = self.compute(z)
        # print("a * (1 -a) = ", a * (1 -a))
        return a*(1-a)


class TanH(Activation):

    def compute(self, Z: np.ndarray) -> np.ndarray:
        """
        Passes Z through a hyperbolic tangent function and returns the values in an np.ndarray
        >>> tan = Tanh()
        >>> l = np.array([-12, 0, 1, 24])
        >>> return tan.compute(l)
        >>> [-1.          0.          0.76159416  1.        ]
        """
    
        return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))

    def derivative(self, gradients: np.ndarray) -> np.ndarray:
        """ """
        pass
        

class ReLU(Activation):

    def compute(self, Z: np.ndarray) -> np.ndarray:
        """
        Passes Z through a rectifier Linear Unit and returns the values in an np.ndarray
        >>> relu = ReLU()
        >>> l = np.array([-12, 0, 1, 24, -2])
        >>> return relu.compute(l))
        >>> [ 0  0  1 24  0]
        """

        return np.where(Z<=0, 0, Z) # equivalent of np.array([0 if x <= 0 else x for x in Z])
    
    def derivative(self, Z: np.ndarray) -> np.ndarray:
        """"""
        return (Z > 0).astype(float)
    
class Softmax(Activation):

    def compute(self, Z: np.ndarray) -> np.ndarray:

        """
        """

        return np.exp(Z) / np.sum(np.exp(Z))

    def derivative(self, gradients: np.ndarray) -> np.ndarray:
        pass
        

