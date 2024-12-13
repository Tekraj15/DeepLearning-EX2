import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output):
        grad_input = grad_output * (self.input > 0)
        return grad_input
