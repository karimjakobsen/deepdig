from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def update(self, learning_rate = 0.01):
        pass
    
class GradientDescent(Optimizer):
    
    """"""
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        
    def update(self, layer) -> None:
        """updates weights and biases in layer instances after backpropagation e.g. lr*gradient
        """
        print("Before update - weights mean:", np.mean(layer.weights), "bias mean:", np.mean(layer.bias))
        print("Gradient - dW mean:", np.mean(layer.dW), "db mean:", np.mean(layer.db))
        layer.weights -= (self.learning_rate * layer.dW)
        layer.bias -= (self.learning_rate * layer.db)
        print("After update - weights mean:", np.mean(layer.weights), "bias mean:", np.mean(layer.bias))
        
    
class SGD(Optimizer):
    """"""
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate
        
    def update(self, layer) -> None:
        """updates weights and biases in layer instances after backpropagation e.g. lr*gradient
        """
        pass
    
class Adam(Optimizer):
    """"""
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate
        
    def update(self, layer) -> None:
        """updates weights and biases in layer instances after backpropagation e.g. lr*gradient
        """
        pass
