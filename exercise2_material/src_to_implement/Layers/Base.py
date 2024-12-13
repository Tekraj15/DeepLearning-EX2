class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.params = {}

    def forward(self, input):
        raise NotImplementedError("Forward pass not implemented.")

    def backward(self, grad_output):
        raise NotImplementedError("Backward pass not implemented.")
