import numpy as np

class NumpyValue:
    """A NumPy-based implementation of the Value class for efficient array operations."""
    
    def __init__(self, data, _children=(), _op='', label=''):
        # Convert input to numpy array if it isn't already
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"NumpyValue(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, NumpyValue) else NumpyValue(other)
        out = NumpyValue(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Gradient flows straight through addition
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, NumpyValue) else NumpyValue(other)
        out = NumpyValue(self.data * other.data, (self, other), '*')
        
        def _backward():
            # Product rule
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = NumpyValue(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = np.tanh(x)
        out = NumpyValue(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        t = np.exp(x)
        out = NumpyValue(t, (self,), 'exp')
        
        def _backward():
            self.grad += t * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """Compute gradients through backpropagation."""
        # Build topological order of all nodes
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        # Go one variable at a time and apply the chain rule
        self.grad = np.ones_like(self.data)  # Initialize with ones for root variable
        for v in reversed(topo):
            v._backward()

def create_random_value(shape):
    """Helper function to create random NumpyValue objects."""
    return NumpyValue(np.random.randn(*shape) * 0.01)

# Example usage
if __name__ == '__main__':
    # Create some values
    x = NumpyValue(2.0)
    y = NumpyValue(3.0)
    
    # Basic operations
    z = x * y + x
    z.backward()
    
    print(f"x: {x.data}, grad: {x.grad}")  # Should show gradient of y + 1
    print(f"y: {y.data}, grad: {y.grad}")  # Should show gradient of x
    print(f"z: {z.data}, grad: {z.grad}")  # Should be 1.0
