from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    @abstractmethod
    def compute(self, z):
        pass
    
    @abstractmethod
    def derivative(self, z):
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
            return self.value
        
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """calculates the derivative of the MSE loss (self.value) and returns it as a float
        """

        return 2*(y_pred - y_true)/y_true.size

class CrossEntropy(Loss):
    def __init__(self):
        self.value = None
        
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        returns mean squared error of all y_true[i]-y_pred[i]
        """

        pass
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """calculates the derivative of the MSE loss (self.value) and returns it as a float
        """

        pass
