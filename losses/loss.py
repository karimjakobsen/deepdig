import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def compute(self, z):
        pass
    
    @abstractmethod
    def loss_gradient(self, z):
        pass

class MSE(Loss):
    def __init__(self):
        self.value = None
        
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        returns mean squared error of all y_true[i]-y_pred[i]
        """

        #assert y_true and y_pred are same shape
        if y_true.shape != y_pred.shape:
            raise ValueError("Expected two np.ndarrays of same shape but got two of different shapes")
        else:
            self.value = np.mean((y_true - y_pred)**2)
            return np.mean((y_true - y_pred)**2)
        
    def loss_gradient(self) -> float:
        """calculates the gradient of the MSE loss (self.value) and returns it as a float
        """

        return

class CrossEntropy(Loss):
    def __init__(self):
        self.value = None
        pass
    
