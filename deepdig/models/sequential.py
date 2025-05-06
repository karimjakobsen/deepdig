from deepdig.optimizers.gradientdescent import Optimizer
from deepdig.losses.loss import Loss
from deepdig.layers.layer import Layer
import numpy as np

class Sequential:
    def __init__(self, layers: list[Layer], optimizer: Optimizer, loss: Loss, epochs = 100):

        self.loss = loss
        self.layers = layers
        self.optimizer = optimizer
        self.cache = None
        self.epochs = epochs

    def build(self):
        pass
    
    def train(self, x: np.ndarray, y_true: np.ndarray):
        #handle epochs here
        for epoch in range(self.epochs):
            self.train_step(x, y_true)

            #print epoch and loss of current opoch
            print(f"Epoch: {epoch+1} of {self.epochs}, Loss: {self.loss.value:.4f}")

        print(f"Prediction:", self.predict(x))
    
    def train_step(self, x: np.ndarray, y_true: np.ndarray):
        """where x is feature vector and y_true is label vector"""

        # 1. forward pass
        y_pred = self.forward(x)

        # 2. compute loss of output from latest forward 
        self.loss.value = self.loss.compute(y_true, y_pred)
        
        # 3. backwards pass
        self.backpropagation(y_pred, y_true)

        # 4. update weights
        for layer in self.layers:
            self.optimizer.update(layer) #update weights in all layers
    

    def forward(self, x: np.ndarray):
        """        
            For the initial pass x is the input. After initial pass x is the output of the previous pass.  
        """
        #initial input stored for use in backpropagation
        self.cache = x
        
        for layer in self.layers:

            #for every iteration x is the output of previous layer passed as input to the new layer
            x = layer.forward(x)
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        predicts an output based on input x. 
        """

        return self.forward(x)
    def backpropagation(self, y_true: np.ndarray, y_pred: np.ndarray):

        
        #loss_gradient dL/da
        #✅  loss derivative w.r.t output activation
        dL_da = self.loss.derivative(y_true, y_pred)

        #loop (resversed) through self.layers
        for i in range(len(self.layers) - 1, -1, -1):

            #current layer
            layer = self.layers[i]
            
            #sigmoid_gradient da/dz
            #✅ derivative of sigmoid function
            da_dz = layer.activation.derivative(layer.z)
            
            #delta_gradient (loss_gradient * sigmoid_gradient)
            #✅ chain rule applied here
            dL_dz = dL_da * da_dz

            # dL/db = sum across batch (axis=1, maintains column shape)
            dL_db = np.sum(dL_dz, axis=1, keepdims=True) ############## <--- need to understand this part better
            print(dL_db) #To see data
            
            #weight_gradients
            #✅ derivative of loss w.r.t weights: dL/dW = delta * a_prev.T
            prev_layer = self.layers[i-1].a if i > 0 else self.cache
                
            dL_dW = dL_dz @ prev_layer.T
            
            layer.dW = dL_dW
            layer.db = dL_db

