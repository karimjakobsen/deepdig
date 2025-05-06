from deepdig.layers.activation import Activation
import numpy as np
import math

class Tanh(Activation):

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
