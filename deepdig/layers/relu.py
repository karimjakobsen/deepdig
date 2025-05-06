from deepdig.layers.activation import Activation
import math
import numpy as np

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
    
    def derivative(self, gradients: np.ndarray) -> np.ndarray:
        """"""
