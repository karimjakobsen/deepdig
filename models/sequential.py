
class Sequential:
    def __init__(self, loss_function: Loss, layers: [Dense]):

        self.loss_function = loss_function
        self.layers = layers
        
        
        
