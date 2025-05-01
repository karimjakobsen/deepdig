import numpy as np
from deepdig.layers.activation import Activation, Sigmoid
class Dense:
    def __init__(self, neurons: int, activation: Activation):

        self.neurons = neurons
        self.activation = activation
        self.weights = None
        self.bias = None
        self.z = None #sore for backprop
        self.a = None #store for backprop
        self.input_dim = None #set a first forward pass

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

activation = Sigmoid()

layer = Dense(4, activation)

x = np.array([[0.4], [-4.2]]) #shape (2,1) (2 samples 1 feature)

out = layer.forward(x)

print(out)
