class GradientDescent:
    def __init__(self, layer: Layer, learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.layer = layer
        
    def update(self) -> None:
        """updates weights and biases in layer instances after backpropagation e.g. lr*gradient
        """
        self.layer.weights -= learning_rate * dW #dW must be passed from Sequential.backpropagation
        self.layer.bias -= learning_rate * db #db must be passed from Sequential.backpropagation
        
    
