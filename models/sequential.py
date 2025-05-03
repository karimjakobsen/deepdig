
class Sequential:
    def __init__(self, loss: Loss, layers: list[Layer], optimizer: Optimizer):

        self.loss = loss
        self.layers = layers
        self.optimizer = optimizer

    def build(self):
        pass
    
    def train():
        pass
    
    def train_step(self, x: np.ndarray, y_true: np.ndarray)
    """where x is feature vector and y_true is label vector"""

    # 1. forward pass
    y_pred = self.forward(x)

    # 2. backwards pass
    self.backpropagation(y_pred, y_true)

    # 3. update weights
    self.optimizer.update(self.layers) #update weights in all layers

    def forward(self, x: np.ndarray):
        """        
            For the initial pass x is the input. After initial pass x is the output of the previous pass.  
        """

        for layer in self.layers:

            #for every iteration x is the output of previous layer passed as input to the new layer
            x = layer.forward(x)

    def backpropagation(self, y_pred: np.ndarray, y_true: np.ndarray):

        
        #loss_gradient dL/da
        self.loss.gradient
        #✅  loss derivative w.r.t output activation

        #loop (resversed) through self.layers
        
            #sigmoid_gradient da/dz
            #✅ derivative of sigmoid function
        
            #delta_gradient (loss_gradient * sigmoid_gradient)
            #✅ chain rule applied here
        
            #weight_gradients
            #✅ derivative of loss w.r.t weights: dL/dW = delta * a_prev.T
        
        #GradientDescent here
        #✅ use gradients to update weights/biases
        
            
        pass
