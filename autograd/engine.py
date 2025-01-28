from graphviz import Digraph
def trace(root): # using dfs to build a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes: # only add each node once
            nodes.add(v)
            for child in v._prev: # traverse each child only once
                edges.add((child, v)) # from child to parent
                build(child)
    build(root)
    return nodes, edges
def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR means left to right
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label = "{%s | %s | data %.2f | grad %.2f}" % (n._op, n.label, n.data, n.grad), shape='record')
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)))
    return dot

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

    def __radd__(self, other): # support cases like "2 + a"
        return self + other

    def __rmul__(self, other): # support cases like "2 * a"
        return self * other

    def __pow__(self, other): # support cases like "a ** 2"
        assert isinstance(other, int) or isinstance(other, float), "Exponent must be int or float"
        out = Value(self.data**other, (self, ), f'**{other}')
        def _backward():
            self.grad += out.grad * other * (self.data**(other-1))
        out._backward = _backward
        return out

    def __truediv__(self, other): # support cases like "a / 2"
        return self * other**-1

    def __rtruediv__(self, other): # support cases like "2 / a"
        return other / self

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
        return outs

class MLP:
    def __init__(self, nin, nouts): # now we have multiple "number of outs"
        sz = [nin] + nouts # nin as the first element
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x