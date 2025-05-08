import numpy as np
from deepdig.layers.activation import Activation, Sigmoid, TanH, ReLU
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass
    
class Dense(Layer):
    def __init__(self, neurons = 4, activation = 'sigmoid'):
        self.neurons = neurons
        self.activation = activation
        self.weights = None
        self.bias = None
        self.dW = None
        self.db = None
        self.z = None #sore for backprop
        self.a = None #store for backprop
        self.input_dim = None #set a first forward pass

    def set(self):
        """Sets activation function"""

        if self.activation == 'sigmoid':
            self.activation = Sigmoid()
        else:
            raise Exception("Invalid input. Valid inputs: 'sigmoid', 'tanh', 'softmax', 'relu' in lower case")
        
    def initialize(self):
        """
        initialize weights and bias
        """
        self.weights = np.random.randn(self.neurons, self.input_dim) * 0.01
        self.bias = np.zeros((self.neurons, 1))

    def forward(self, x):
        """pass x through layer
        """

        # initalize weigts/bias
        if self.weights is None:
            self.input_dim = x.shape[0] #specify input dimension of layer
            self.initialize()
            
        self.z = np.dot(self.weights, x)+self.bias
        self.a = self.activation.compute(self.z)
        return self.a
    
