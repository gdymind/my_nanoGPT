from engine import *
import numpy as np
import random

class Neuron:
    def __init__(self, nin): # nin: number of inputs
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        # w * x + b
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts): # now we have multiple "number of outs"
        sz = [nin] + nouts # nin as the first element
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# define the main function
if __name__ == '__main__':
    xs = [
        [2.0, 3.0, 1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    nouts = [4, 4, 1]
    mlp = MLP(3, nouts)
    ypred = [mlp(x) for x in xs]

    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    print(loss)

    # optimize the parameters manually without using an optimizer
    params = mlp.parameters()
    for i in range(20):
        # forward pass
        ypred = [mlp(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        # backward pass
        for p in params:
            p.grad = 0.0
        loss.backward()

        # update parameters
        for p in params:
            p.data -= 0.01 * p.grad

        print(f"step {i}: loss: {loss.data:.8f}")