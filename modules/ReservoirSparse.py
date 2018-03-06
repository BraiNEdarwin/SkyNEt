"""
Reservoir.py
~~~~~~~~~~
A program designed to represent a reservoir computer, that
can be trained to output a desired signal.
"""

# Libraries
# Third-party libraries
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg


class Network(object):

    def __init__(self, sizes, inputscaling, spectralradius, weightdensity):
        """
        Initializing a reservoir of the proper size.
        """
        # Initialize a reservoir of the proper size
        self.sizes = sizes
        self.state = np.zeros((sizes[1], 1))
        # Build and scale input weights, output and reservoir weights
        self.input_weights = -1 + 2 * np.random.rand(sizes[1], sizes[0])
        self.input_weights = inputscaling * self.input_weights / \
            np.linalg.norm(max(self.input_weights))
        self.reservoir_weights = 2 * \
            sparse.rand(sizes[1], sizes[1], density=weightdensity)
        self.reservoir_weights.data += -1
        self.eval, self.evecs = splinalg.eigs(
            self.reservoir_weights, k=1, which='LM')
        print("Largest eigenvalue magnitude before normalizing: " + str(
            self.eval))
        self.reservoir_weights = (spectralradius * self.reservoir_weights) / \
            np.linalg.norm(self.eval)
        self.output_weights = np.random.randn(sizes[2], sizes[1])
        self.collect_state = np.empty((0, sizes[1]))
        self.eval, self.evecs = splinalg.eigs(
            self.reservoir_weights, k=1, which='LM')
        print("Spectral radius of the weight matrix is " +
              str(np.linalg.norm(self.eval)))
        self.reservoir_weights = self.reservoir_weights.toarray()

    def update_reservoir(self, input_val):
        # Build the input for each reservoir node
        input_arr = np.dot(self.input_weights, input_val)
        activation = np.dot(self.reservoir_weights, self.state) + input_arr
        # Calculate the new state
        self.state = tanh(activation)
        self.reservoir_output = tanh(
            np.dot(self.output_weights, self.state))
        # Collect the states
        self.collect_state = np.append(
            self.collect_state, self.state.transpose(), axis=0)

    def train_reservoir_pseudoinv(self, target, skipstates):
        self.trained_weights = np.transpose(np.dot(
            np.linalg.pinv(self.collect_state[skipstates:]), target[skipstates:]))
        trained_output = np.dot(self.trained_weights,
                                self.collect_state[skipstates:].transpose())
        return trained_output

    def train_reservoir_ridgereg(self, target, alpha, skipstates):
        R = np.dot(self.collect_state[skipstates:].transpose(
        ), self.collect_state[skipstates:])
        P = np.dot(self.collect_state[skipstates:].transpose(), target[skipstates:])
        Id = np.identity(self.sizes[1])
        self.trained_weights = np.dot(np.linalg.inv(R + alpha ** 2 * Id), P)
        trained_output = np.dot(self.trained_weights,
                                self.collect_state[skipstates:].transpose())
        return trained_output


# Miscellaneous functions

def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z):
    return np.tanh(z)
