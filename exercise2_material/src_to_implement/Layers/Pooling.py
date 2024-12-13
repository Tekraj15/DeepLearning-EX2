import numpy as np

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channels, input_height, input_width = input_tensor.shape
        pool_height, pool_width = self.pooling_shape
        stride_h, stride_w = self.stride_shape
        output_height = (input_height - pool_height) // stride_h + 1
        output_width = (input_width - pool_width) // stride_w + 1
        output = np.zeros((batch_size, channels, output_height, output_width))
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start, h_end = i * stride_h, i * stride_h + pool_height
                        w_start, w_end = j * stride_w, j * stride_w + pool_width
                        output[b, c, i, j] = np.max(input_tensor[b, c, h_start:h_end, w_start:w_end])
        return output

    def backward(self, error_tensor):
        # Implement gradients for pooling layer
        pass
