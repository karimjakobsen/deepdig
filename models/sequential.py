
class Sequential:
    def __init__(self, loss: Loss, layers: list[Layer], optimizer: str):

        self.loss = loss
        self.layers = layers
        self.opt = None

    def build(self):
        pass

    def forward(self):
        pass

    def backpropagation(self):
        
        #loss_gradient dL/da
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
