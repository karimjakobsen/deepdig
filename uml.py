import plantuml
import os

# PlantUML code for the neural network library
uml_code = """
@startuml deepdig_uml

' --- Layers ---
class Dense {
  - weights: ndarray
  - biases: ndarray
  + __init__(units: int, input_dim: int?)
  + forward(X: ndarray) -> ndarray
  + backward(grad: ndarray) -> ndarray
}

class ReLU {
  + forward(X: ndarray) -> ndarray
  + backward(grad: ndarray) -> ndarray
}

' --- Losses ---
class MSE {
  + compute(y_true: ndarray, y_pred: ndarray) -> float
  + gradient(y_true: ndarray, y_pred: ndarray) -> ndarray
}

class CrossEntropy {
  + compute(y_true: ndarray, y_pred: ndarray) -> float
  + gradient(y_true: ndarray, y_pred: ndarray) -> ndarray
}

' --- Optimizers ---
class SGD {
  - lr: float
  + __init__(lr: float)
  + update(layer: Dense) -> None
}

class Adam {
  - lr: float
  - m: ndarray
  - v: ndarray
  + __init__(lr: float, beta1: float, beta2: float)
  + update(layer: Dense) -> None
}

' --- Models ---
class Sequential {
  - layers: List[Layer]
  + add(layer: Layer) -> None
  + fit(X: ndarray, y: ndarray, epochs: int, loss: Loss, optimizer: Optimizer) -> None
  + predict(X: ndarray) -> ndarray
}

' --- Relationships ---
Sequential "1" *-- "1..*" Dense : Contains
Sequential "1" *-- "1..*" ReLU : Contains
Dense <|-- Layer : Implements
ReLU <|-- Activation : Implements
MSE <|-- Loss : Implements
CrossEntropy <|-- Loss : Implements
SGD <|-- Optimizer : Implements
Adam <|-- Optimizer : Implements

' --- Interfaces (Abstract Classes) ---
interface Layer {
  + forward(X: ndarray) -> ndarray
  + backward(grad: ndarray) -> ndarray
}

interface Activation {
  + forward(X: ndarray) -> ndarray
  + backward(grad: ndarray) -> ndarray
}

interface Loss {
  + compute(y_true: ndarray, y_pred: ndarray) -> float
  + gradient(y_true: ndarray, y_pred: ndarray) -> ndarray
}

interface Optimizer {
  + update(layer: Layer) -> None
}

@enduml
"""

# Write UML code to a file
with open("deepdig_uml.puml", "w") as file:
    file.write(uml_code)

# Use PlantUML server to generate the diagram
plantuml_server = plantuml.PlantUML(url="http://www.plantuml.com/plantuml/png/")
plantuml_server.processes_file("deepdig_uml.puml", outfile="deepdig_uml.png")

# Convert PNG to PDF (requires external tool like ImageMagick or manual conversion)
# Note: Uncomment the following if ImageMagick is installed locally
# os.system("convert deepdig_uml.png deepdig_uml.pdf")

print("UML diagram generated as deepdig_uml.png. Convert to PDF manually or use ImageMagick.")
