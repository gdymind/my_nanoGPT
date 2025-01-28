from autograd.numpy_engine import NumpyValue
import numpy as np

class NumpyNeuron:
    def __init__(self, nin):  # nin: number of inputs
        # Initialize weights and bias as NumpyValue objects with array data
        self.w = NumpyValue(np.random.randn(nin) * 0.01)
        self.b = NumpyValue(np.random.randn() * 0.01)
    
    def __call__(self, x):
        # Handle inputs that are already NumpyValue objects or numpy arrays
        if isinstance(x, NumpyValue):
            x_data = x.data
        elif isinstance(x, np.ndarray):
            x_data = x
        else:
            # Convert input to numpy array if it's not already
            x_data = np.array([x] if np.isscalar(x) else x, dtype=np.float64)
        
        # Compute w * x + b using dot product for efficiency
        act = NumpyValue(np.dot(self.w.data, x_data)) + self.b
        return act

class NumpyLayer:
    def __init__(self, nin, nout):
        # Create a matrix of weights (nout x nin) for efficient matrix multiplication
        self.neurons = [NumpyNeuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        # If input is a list of NumpyValues, extract their data
        if isinstance(x, list) and all(isinstance(v, NumpyValue) for v in x):
            x = [v.data for v in x]
        # Process all neurons in parallel
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class NumpyMLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [NumpyLayer(sz[i], sz[i+1]) for i in range(len(nouts))]
        
    def parameters(self):
        """Return all parameters of the network."""
        params = []
        for layer in self.layers:
            for neuron in layer.neurons:
                params.extend([neuron.w, neuron.b])
        return params
    
    def __call__(self, x):
        # Forward pass through the network
        for layer in self.layers:
            x = layer(x)
        return x

def test_network():
    """Test the neural network with some sample data."""
    # Create input data
    x = np.array([1.0, 2.0, 3.0])
    
    # Create a small network
    nin = 3
    nouts = [4, 4, 1]  # 3 -> 4 -> 4 -> 1
    
    # Initialize the network
    mlp = NumpyMLP(nin, nouts)
    
    # Forward pass
    output = mlp(x)
    
    # Compute gradients
    output.backward()
    
    return output, mlp

if __name__ == '__main__':
    # Run test
    output, mlp = test_network()
    print(f"Network output: {output.data}")
    
    # Print some gradients
    params = mlp.parameters()
    print("\nFirst layer, first neuron weights gradient:")
    print(params[0].grad)  # First neuron's weights
    print("\nFirst layer, first neuron bias gradient:")
    print(params[1].grad)  # First neuron's bias
