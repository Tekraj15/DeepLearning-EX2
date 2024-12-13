import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def forward(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    def backward(self, grad_output):
        # SoftMax backward pass
        raise NotImplementedError("Backward pass for SoftMax is task-specific.")
