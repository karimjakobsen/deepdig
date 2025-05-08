from abc import ABC, abstractmethod

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
        layer.weights -= self.learning_rate * layer.dW 
        layer.bias -= self.learning_rate * layer.db 
    
    
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
