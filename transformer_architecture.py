import numpy as np
import pandas as pd
from cnn_architecture import SoftmaxLayer

class Transformer:
    pass

class SelfAttention:
    def __init__(self, dimension: list[int]):
        self.dimension = dimension
        self.rng = np.random.default_rng()
        self.Wq = np.rint(self.rng.normal(size = tuple(dimension))).astype(np.float32)
        self.Wk = np.rint(self.rng.normal(size = tuple(dimension))).astype(np.float32)
        self.Wv = np.rint(self.rng.normal(size = tuple(dimension))).astype(np.float32)

    def forward(self, x: np.ndarray):
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv
        softmax = SoftmaxLayer()

        attentionScores = softmax.forward((q @ k.T / np.sqrt(self.dimension[0]))) @ v
        self.A = attentionScores

        return attentionScores

    def backward(self, upstream_gradient, x):
        softmax = SoftmaxLayer()
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv

        dO = upstream_gradient
        dA = dO @ v
        dV = self.A.T @ dO

        dS = softmax.backwards(dA)
        dQ = dS @ k / np.sqrt(self.dimension[0])
        dK = dS @ q / np.sqrt(self.dimension[0])

        self.dWq = x.T @ dQ
        self.dWk = x.T @ dK
        self.dWv = x.T @ dV

        return dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T # DX as computed by the sum of the gradients of the weight vectors.

class MultiHeadAttention:
    # TODO Implement Masked Self Attention
    # Precision is lower with MultiHead Attention since the output dimension of each Attention Head is lower.
    def __init__(self, num_heads: int, dimensions: int, masked = False):
        self.inputDimensions = dimensions
        self.outputDimensions = dimensions // num_heads
        self.numHeads = num_heads
        self.Wo = np.rint(self.rng.normal(size = tuple(dimensions, dimensions))).astype(np.float32) / np.sqrt(dimensions)
        self.masked = masked

        self.attentionHeads = []
        for i in range(num_heads):
            self.attentionHeads.append(SelfAttention([self.inputDimensions, self.outputDimensions]))

    def forward(self, x):
        outputs = []
        for i in range(len(self.attentionHeads)):
            head = self.attentionHeads[i]
            out = head.forward(x)
            outputs.append(out)

        self.H = np.concatenate(outputs, axis = -1)
        return np.concatenate(outputs, axis = -1) @ self.Wo

    def backward(self, upstream_gradient):

        self.dWo = self.H.T @ upstream_gradient
        dH = upstream_gradient @ self.Wo.T
 
        dHeads = np.split(dH, self.numHeads, axis=-1)
        
        dx = np.zeros(self.attentionHeads[0].x.shape)
        for i in range(self.numHeads):
            head = self.attentionHeads[i]
            dh = dHeads[i]
            dx += head.backward(dh)
        return dx

class LayerNorm:
    # TODO Fix this implementation - figure out how to use NP for this better.
    def __init__(self):
        pass

    def forward(x):
        newArray = []
        for i in range(x.shape[0]):
            mean = np.mean(x[i])
            std = np.std(x[i])

            newArray.append(x[i] - mean / std)

        return np.array(newArray)

class RELU:
    def __init__(self):
        pass

    def forward(self, x):
        return max(0, x)


class TransformerEncoder:
    # TODO Add Positional Encoding Layer. - Add Bias term to FFN also.
    def __init__(self, x: np.ndarray, num_heads: int, dimensions: int):
        self.x = x
        self.numHeads = num_heads
        self.dimensions = dimensions
        self.multiHead = MultiHeadAttention(self.numHeads, self.dimensions)
        self.FFN_W = np.rint(self.rng.normal(size = tuple(dimensions, dimensions))).astype(np.float32)
        self.FFN_B = np.rint(self.rng.normal(size = tuple(dimensions, ))).astype(np.float32)

    def forward(self):
        beforeGate = self.x
        encodedInformation = self.multiHead.forward(self.x)
        addedResidual = encodedInformation + beforeGate
        layerNorm = LayerNorm()
        normalized = layerNorm.forward(addedResidual)
        relu = RELU()
        FFN = relu(normalized @ self.FFN_W + self.FFN_B)
        addedResidual2 = normalized + FFN
        normalized2 = layerNorm.forward(addedResidual2)
        return normalized2

    def backward(self):

        pass


class TransformerDecoder:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass



    