import numpy as np
from Layers import FullyConnected, Conv, Flatten, Pooling, SoftMax
from Optimization import Loss, Optimizers
from Optimization.Loss import CrossEntropyLoss

class NeuralNetwork:
    def __init__(self, optimizer, loss_fn=CrossEntropyLoss()):
        """
        Initialize the Neural Network with the given optimizer and loss function.
        """
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.layers = []
        self.loss_history = []
        self.data_layer = None
        self.loss_layer = loss_fn

    def append_layer(self, layer):
        """
        Add a layer to the neural network. If the layer is trainable, the optimizer is copied and set for the layer.
        """
        if hasattr(layer, 'trainable') and layer.trainable:
            layer.optimizer = self.optimizer  # Assign the optimizer to trainable layers
        self.layers.append(layer)

    def forward(self):
        """
        Perform forward propagation through all layers in the network.
        """
        input_tensor, label_tensor = self.data_layer.next()  # Assuming data_layer provides batches
        output_tensor = input_tensor

        # Pass the input through each layer
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)

        # Compute the loss at the end
        loss = self.loss_layer.forward(output_tensor, label_tensor)
        self.loss_history.append(loss)
        return loss

    def backward(self):
        """
        Perform backpropagation through all layers in the network.
        """
        output_tensor, label_tensor = self.data_layer.next()  # Assuming data_layer provides batches
        grad_output = self.loss_layer.backward(label_tensor)

        # Backpropagate through each layer in reverse order
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
            if hasattr(layer, 'update'):
                layer.update()

    def train(self, iterations):
        """
        Train the network for a number of iterations.
        """
        for i in range(iterations):
            loss = self.forward()
            self.backward()
            if i % 100 == 0:  # Optionally print progress every 100 iterations
                print(f"Iteration {i}/{iterations}, Loss: {loss}")

    def test(self, input_tensor):
        """
        Test the neural network on a given input tensor.
        """
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)

        return output
