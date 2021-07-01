import numpy as np
import math


class Network:
    eta = 0.001

    def __init__(self, topology):
        self.topology = topology
        self.weights = {}
        for index in range(len(self.topology) - 1):
            weights_mat = []
            for i in range(self.topology[index]):
                weights = []
                for j in range(self.topology[index + 1]):
                    weights.append(np.random.normal())
                weights_mat.append(weights)
            biases = []
            for j in range(self.topology[index + 1]):
                biases.append(np.random.normal())
            weights_mat.append(biases)
            self.weights[index] = np.matrix(weights_mat)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x * 1.0))

    def dSigmoid(self, x):
        return x * (1.0 - x)

    def setInput(self, inputs):
        self.layer_outs = {}
        inputs = np.append(inputs, 1)
        self.layer_outs[0] = inputs

    def feedForword(self):
        for layer in range(1, len(self.topology)):
            layer_out = self.sigmoid(np.matmul(np.array(self.layer_outs[layer - 1]), self.weights[layer - 1]))
            layer_out = np.copy(layer_out[0])
            if layer != (len(self.topology) - 1):
                layer_out = np.append(layer_out[0], 1)
            self.layer_outs[layer] = layer_out
            
    def backPropagate(self, target):
        next_layer_err = target - self.layer_outs[len(self.topology) - 1]
        for index in reversed(range(len(self.topology) - 1)):
            next_layer_grad = np.multiply(next_layer_err, self.dSigmoid(self.layer_outs[index + 1]))
            prv_layer_out = self.layer_outs[index]
            prv_layer_out.shape = (1, self.topology[index] + 1)
            derivative_weights = Network.eta * np.outer(prv_layer_out, next_layer_grad)
            if derivative_weights.shape != self.weights[index].shape:
                derivative_weights = np.delete(derivative_weights, -1, 1)
                next_layer_grad = np.delete(next_layer_grad, -1, 1)
            self.weights[index] += derivative_weights
            next_layer_err = np.matmul(self.weights[index], next_layer_grad.T)
            next_layer_err = next_layer_err.T

    def getError(self, target):
        e = target - self.layer_outs[len(self.topology) - 1]
        err = np.sum(e ** 2)
        err = err / len(target)
        err = math.sqrt(err)
        return err

    def getResults(self):
        results = np.copy(self.layer_outs[len(self.topology) - 1][0])
        return results
    
    def getThResults(self):
        results = np.copy(self.layer_outs[len(self.topology) - 1][0])
        results = [float(int(i/np.max(results))) for i in results]
        return results