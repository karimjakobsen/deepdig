from deepdig.optimizers.optimizer import Optimizer, GradientDescent
from deepdig.losses.loss import Loss, MSE
from deepdig.layers.dense import Layer, Dense
import numpy as np

class Sequential:
    def __init__(self, layers: list[Layer], optimizer: str = "gradient_descent", learning_rate: float = 0.01, loss: str = "mse", epochs = 100):

        self.loss = loss
        self.layers = layers
        self.optimizer = optimizer
        self.cache = None
        self.epochs = epochs
        self.learning_rate = learning_rate

    def build(self):
        """
        Sets the correct loss function based on parameter self.loss
        Sets the correct optimizer algorithm based on parameter self.optimizer
        Sets the correct activation function for each Layer.
        """

        
        if not type(self.learning_rate) == float:
            raise Exception("learning_rate must be of type float")

        # set loss function for model
        if self.loss == "mse":
            self.loss = MSE()
        elif self.loss == "cross_entropy":
            self.loss = CrossEntropy()
        else:
            raise Exception("Valid loss functions: 'mse', 'cross_entropy'. Specify in lowercase")

        # set optimizer for model
        if self.optimizer == "gradient_descent":
            self.optimizer = GradientDescent(self.learning_rate)
        elif self.optimizer == "adam":
            self.optimizer = Adam(self.learning_rate)
        else:
            raise Exception("Valid optimizers: 'gradient_descent', 'adam', 'sgd'. Specify in lowercase")

        # set activation function for each layer
        for layer in self.layers:
            layer.set()
                
    
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

        print('predicted:', np.shape(y_pred))
        print('true:', np.shape(y_true))

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

