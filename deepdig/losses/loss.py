import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def compute(self, z):
        pass
    
    @abstractmethod
    def derivative(self, z):
        pass


    
