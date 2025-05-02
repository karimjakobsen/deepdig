class GradientDescent:
    def __init__(self, learning_rate = 0.01, layers):
        self.learning_rate = learning_rate

    def update(self, learning_rate):
        """updates weights and biases in layer instances after backpropagation e.g. lr*gradient
        """

        for layer in layers:
            layer.weights -=
