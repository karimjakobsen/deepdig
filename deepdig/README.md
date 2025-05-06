Simple neural network package.
Project made to better understand deep learning architectures and underlying logic.

Install:
pip install git+https://github.com/karimjakobsen/deepdig.git


Basic Sequential model example:
====================================================================================================
import deepdig as dd
from deepdig.optimizers.gradientdescent import GradientDescent
from deepdig.losses.mse import MSE
from deepdig.layers.sigmoid import Sigmoid
from deepdig.layers.dense import Dense
from deepdig.models.sequential import Sequential


# Create model
model = Sequential([Dense(neurons=64, activation=Sigmoid()),
                    Dense(neurons=64, activation=Sigmoid())],
                   optimizer=GradientDescent(),
                   epochs=100, loss=MSE())
====================================================================================================

