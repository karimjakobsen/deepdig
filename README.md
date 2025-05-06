deepdig is a deep learning library
It is being developed as an exercise to better understand core DL concepts
At this point there is only a basic sequential model with dense layers available.

=============== PROJECT STRUCTURE ======================
deepdig/
├── layers/           # Layer implementations
│   ├── dense.py      # Dense (fully connected) layer
│   └── activation.py # ReLU, Sigmoid
├── losses/           # Loss functions
│   ├── mse.py
│   └── cross_entropy.py (NOT IMPLEMTED)
├── optimizers/       # Optimization algorithms
│   ├── sgd.py        (NOT IMPLEMTED)
│   └── adam.py       (NOT IMPLEMTED) 
    └── gradient_descent.py 
├── models/           # Network architectures
│   └── sequential.py # Sequential model
└── utils/            # Helpers (NOT IMPLEMENTED)
    ├── initializers.py # Weight initialization
    └── metrics.py     # Accuracy, precision (NOT IMPLEMENTED)


Designing a neural network:

X -> h1 -> h2 -> output

X = [x1, x2, x3]
h1 = sigmoid(X*w1+b1) = a1
h2 = sigmoid(a1*w2+b2) = a2 = out

y = [3.32, 0.25, 0.89, 2.05, 1.1]

    
