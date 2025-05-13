from deepdig.optimizers.optimizer import Optimizer

class GradientDescent(Optimizer):
    """"""
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate
        
    def update(self, layer) -> None:
        """updates weights and biases in layer instances after backpropagation e.g. lr*gradient
        """
        print("layer.bias: ", layer.bias.shape, "layer.db: ", layer.db.shape)
        layer.weights -= self.learning_rate * layer.dW 
        layer.bias -= self.learning_rate * layer.db
    
    
