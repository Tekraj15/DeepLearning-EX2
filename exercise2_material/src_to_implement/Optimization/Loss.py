import numpy as np

class Loss:
    def forward(self, predicted, actual):
        """
        Compute the loss value given predictions and actual values.

        Parameters:
        - predicted: Predicted values (numpy array)
        - actual: Actual values (numpy array)

        Returns:
        - Loss value
        """
        raise NotImplementedError("Forward method not implemented in base Loss class.")
    
    def backward(self, predicted, actual):
        """
        Compute the gradient of the loss with respect to predictions.

        Parameters:
        - predicted: Predicted values (numpy array)
        - actual: Actual values (numpy array)

        Returns:
        - Gradient of the loss (numpy array)
        """
        raise NotImplementedError("Backward method not implemented in base Loss class.")

class MeanSquaredError(Loss):
    def forward(self, predicted, actual):
        """
        Compute the mean squared error loss.
        """
        return np.mean((predicted - actual) ** 2)

    def backward(self, predicted, actual):
        """
        Compute the gradient of mean squared error loss.
        """
        return 2 * (predicted - actual) / predicted.size

class CrossEntropyLoss(Loss):
    def forward(self, predicted, actual):
        """
        Compute the cross-entropy loss.

        Parameters:
        - predicted: Predicted probabilities (numpy array)
        - actual: Actual class labels (numpy array)

        Returns:
        - Cross-entropy loss value
        """
        epsilon = 1e-12  # To avoid log(0)
        predicted = np.clip(predicted, epsilon, 1. - epsilon)
        return -np.sum(actual * np.log(predicted)) / actual.shape[0]

    def backward(self, predicted, actual):
        """
        Compute the gradient of cross-entropy loss.
        """
        epsilon = 1e-12
        predicted = np.clip(predicted, epsilon, 1. - epsilon)
        return -actual / predicted
