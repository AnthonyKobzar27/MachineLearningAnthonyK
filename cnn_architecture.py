import numpy as np
import pandas as pd

class CNN:
    def __init__(self):
        pass


class ConvolutionalBlock:
    def __init__(self, kernel_size: list[int], feature_maps: int):
        self.kernel_size = kernel_size
        self.feature_maps = feature_maps
        self.rng = np.random.default_rng()
        self.kernels = np.rint(self.rng.normal(size = (feature_maps, *kernel_size))).astype(np.float32) # Float 32 means we are storing the floats in IEE 754 format, 1 signed bit, 8 exponent bits, 23 mantissa bits.

    def forward(self, x: np.ndarray) -> np.array: # Assuming that x is just a 2 dimensional array for now - TODO complete this.
        outputMaps = []
        for kernel in self.kernels:
            outputRows = x.shape[0] - kernel.shape[0] + 1
            outputCols = x.shape[1] - kernel.shape[1] + 1
            currMap = [[0] * (outputCols) for _ in range(outputRows)]
            for i in range(outputRows):
                for j in range(outputCols):
                    currSum = 0
                    for a in range(kernel.shape[0]):
                        for b in range(kernel.shape[1]):
                            currSum += kernel[a][b] * x[i + a][j + b]

                    currMap[i][j] = currSum

            outputMaps.append(currMap)

        return np.array(outputMaps)

    def backward(self, upstream_gradient: np.ndarray, x: np.ndarray) -> np.ndarray:
        gradient_maps = []
        for k in range(self.feature_maps):

            gradient_w = [[0] * self.kernel_size[1] for _ in range(self.kernel_size[0])]
            outputRows = x.shape[0] - self.kernel_size[0] + 1
            outputCols = x.shape[1] - self.kernel_size[1] + 1


            for i in range(outputRows):
                for j in range(outputCols):

                    for a in range(self.kernel_size[0]):
                        for b in range(self.kernel_size[1]):
                            gradient_w[a][b] += upstream_gradient[k][i][j] * x[i + a][j + b] # This is right.

            gradient_maps.append(gradient_w)

        # Still need to update Gradients TODO

        return np.array(gradient_maps)

class DenseLayer:
    def __init__(self, dimensions: list[int]):
        self.dimensions = dimensions
        self.rng = np.random.default_rng()
        self.weights = np.rint(self.rng.normal(size = tuple(dimensions[::-1]))).astype(np.float32)
        self.biases = np.rint(self.rng.normal(size = (dimensions[1],))).astype(np.float32)

    def forward(self, x: np.ndarray):
        return self.weights @ x + self.biases

    def backwards(self, upstream_gradients: np.ndarray, x:np.ndarray):
        return upstream_gradients * x, upstream_gradients


class SoftmaxLayer:
    def __init__(self):
        pass

    def forward(self, logits: list[float]):
        returnedList = []
        for logit in logits:
            value = np.exp(logit) / np.sum([np.exp(myLogit) for myLogit in logits])
            returnedList.append(value)

        return returnedList
        

    def backwards(self, upstream_gradients: list[float], logits: list[float]):
        probabilities = self.forward(logits)
        outputs = [0] * len(logits)
        for i in range(len(upstream_gradients)):
            for j in range(len(probabilities)):
                if i == j:
                    outputs[j] += upstream_gradients[i] * (probabilities[j] * (1 - probabilities[j]))
                else:
                    outputs[j] += upstream_gradients[i] * (-probabilities[i] * probabilities[j])

        return outputs









        








                
        

