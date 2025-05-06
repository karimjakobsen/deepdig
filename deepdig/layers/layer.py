from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def backward(self, gradient):
        pass
