import math
from vis import draw_dot
import numpy as np
import random

class Value:
    def __init__(self, data, _children=(), _op = '', label = ''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None # default: no action for leaf nodes
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self): # for printing
        return f"Value(data={self.data})"

    def __add__(self, other): # a + b
        other = other if isinstance(other, Value) else Value(other) # support cases like "a + 2"
        out = Value(self.data + other.data, (self, other), '+')
        def _backward(): # set self.grad and other.grad based on out.grad
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out

    def __neg__(self): # -a
        return self * -1
    
    def __sub__(self, other): # a - b
        return self + (-other)

    def __mul__(self, other): # a * b
        other = other if isinstance(other, Value) else Value(other) # support cases like "a * 2"
        out = Value(self.data * other.data, (self, other), '*')
        def _backward(): # set self.grad and other.grad based on out.grad
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __pow__(self, other): # support cases like "a ** 2"
        assert isinstance(other, int) or isinstance(other, float), "Exponent must be int or float"
        out = Value(self.data**other, (self, ), f'**{other}')
        def _backward():
            self.grad += out.grad * other * (self.data**(other-1))
        out._backward = _backward
        return out

    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1) / (math.exp(2*n) + 1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += out.grad * (1 - t**2)
        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self,), 'relu')
        def _backward():
            self.grad += out.grad * (out.data > 0)
        out._backward = _backward
        return out

    def __truediv__(self, other): # support cases like "a / 2"
        return self * other**-1

    def __radd__(self, other): # support cases like "2 + a"
        return self + other

    def __rmul__(self, other): # support cases like "2 * a"
        return self * other

    def __rtruediv__(self, other): # support cases like "2 / a"
        return other / self
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()