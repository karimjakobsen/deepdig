from optimizer import Optimizer

class Adam(Optimizer):
        """"""
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate
        
    def update(self, layer) -> None:
        """updates weights and biases in layer instances after backpropagation e.g. lr*gradient
        """
        pass
