from engine import *
import numpy as np

class Neuron:
    def __init__(self, nin): # nin: number of inputs
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        # w * x + b
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return out

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:
    def __init__(self, nin, nouts): # now we have multiple "number of outs"
        sz = [nin] + nouts # nin as the first element
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# define the main function
if __name__ == '__main__':
    x = [1.0, 2.0, 3.0]
    nin = 3
    nouts = [4, 5, 6, 1]
    mlp = MLP(nin, nouts)
    print(mlp(x))