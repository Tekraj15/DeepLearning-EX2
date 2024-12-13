import numpy as np

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        """
        Initialize the Conv layer.

        Parameters:
        - stride_shape: Tuple of strides for convolution (e.g., (1,) for 1D or (1, 1) for 2D)
        - convolution_shape: Shape of the convolution filter (e.g., (channels, height, width) for 2D)
        - num_kernels: Number of kernels (filters)
        """
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        # Initialize placeholders for weights, bias, and gradients
        self.weights = None
        self.bias = None
        self.gradient_weights = None
        self.gradient_bias = None

        self.input_tensor = None  # Store the input tensor for backward pass

    def initialize(self, weights_initializer, bias_initializer):
        """
        Initialize weights and biases using the provided initializers.
        """
        if len(self.convolution_shape) == 2:  # 1D convolution
            self.weights = weights_initializer.initialize(
                (self.num_kernels, self.convolution_shape[0], self.convolution_shape[1])
            )
        elif len(self.convolution_shape) == 3:  # 2D convolution
            self.weights = weights_initializer.initialize(
                (self.num_kernels, *self.convolution_shape)
            )
        else:
            raise ValueError("Invalid convolution shape: must be 1D or 2D.")
        self.bias = bias_initializer.initialize((self.num_kernels,))
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)

    def forward(self, input_tensor):
        """
        Perform the forward pass for the Conv layer.

        Parameters:
        - input_tensor: Input tensor (e.g., [batch_size, channels, height, width] for 2D)

        Returns:
        - Output tensor after applying the convolution.
        """
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]

        if len(input_tensor.shape) == 3:  # 1D input
            input_channels, input_length = input_tensor.shape[1:]
            kernel_length = self.weights.shape[2]
            stride = self.stride_shape[0]
            output_length = (input_length - kernel_length) // stride + 1
            output = np.zeros((batch_size, self.num_kernels, output_length))

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for i in range(output_length):
                        start = i * stride
                        end = start + kernel_length
                        output[b, k, i] = np.sum(
                            input_tensor[b, :, start:end] * self.weights[k]
                        ) + self.bias[k]

        elif len(input_tensor.shape) == 4:  # 2D input
            input_channels, input_height, input_width = input_tensor.shape[1:]
            kernel_height, kernel_width = self.weights.shape[2:]
            stride_h, stride_w = self.stride_shape
            output_height = (input_height - kernel_height) // stride_h + 1
            output_width = (input_width - kernel_width) // stride_w + 1
            output = np.zeros((batch_size, self.num_kernels, output_height, output_width))

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for i in range(output_height):
                        for j in range(output_width):
                            h_start, h_end = i * stride_h, i * stride_h + kernel_height
                            w_start, w_end = j * stride_w, j * stride_w + kernel_width
                            output[b, k, i, j] = np.sum(
                                input_tensor[b, :, h_start:h_end, w_start:w_end] * self.weights[k]
                            ) + self.bias[k]
        else:
            raise ValueError("Unsupported input tensor shape.")

        return output

    def backward(self, error_tensor):
        """
        Perform the backward pass for the Conv layer.

        Parameters:
        - error_tensor: Gradients from the next layer.

        Returns:
        - Gradient with respect to the input tensor.
        """
        batch_size = self.input_tensor.shape[0]
        grad_input = np.zeros_like(self.input_tensor)

        if len(self.input_tensor.shape) == 3:  # 1D convolution
            input_channels, input_length = self.input_tensor.shape[1:]
            kernel_length = self.weights.shape[2]
            stride = self.stride_shape[0]

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for i in range(error_tensor.shape[2]):
                        start = i * stride
                        end = start + kernel_length

                        grad_input[b, :, start:end] += self.weights[k] * error_tensor[b, k, i]
                        self.gradient_weights[k] += self.input_tensor[b, :, start:end] * error_tensor[b, k, i]

                    self.gradient_bias[k] += np.sum(error_tensor[b, k, :])

        elif len(self.input_tensor.shape) == 4:  # 2D convolution
            input_channels, input_height, input_width = self.input_tensor.shape[1:]
            kernel_height, kernel_width = self.weights.shape[2:]
            stride_h, stride_w = self.stride_shape

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for i in range(error_tensor.shape[2]):
                        for j in range(error_tensor.shape[3]):
                            h_start, h_end = i * stride_h, i * stride_h + kernel_height
                            w_start, w_end = j * stride_w, j * stride_w + kernel_width

                            grad_input[b, :, h_start:h_end, w_start:w_end] += self.weights[k] * error_tensor[b, k, i, j]
                            self.gradient_weights[k] += self.input_tensor[b, :, h_start:h_end, w_start:w_end] * error_tensor[b, k, i, j]

                    self.gradient_bias[k] += np.sum(error_tensor[b, k, :, :])

        return grad_input
