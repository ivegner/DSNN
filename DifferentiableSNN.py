import tensorflow as tf
import numpy as np
from collections import deque

tf.enable_eager_execution()

ACTIVATION_HISTORY_LEN = 10

# shape = (width, height, depth)
# n_in, n_out = number of inputs/outputs
class DSNN:
    def __init__(self, shape, n_in, n_out):
        self.shape = [d for d in shape if d > 1]  # no empty dimensions
        self.n_dims = len(self.shape)
        self.multipliers_net = tf.get_variable(
            "multipliers", shape=(1, *self.shape, 1), initializer=tf.random_normal_initializer()
        )

        self.reset_activations()
        # first row for inputs, last row for outputs. TODO: input and output on different heights.
        input_cell_gap = shape[0] // (n_in + 1)
        self.input_indices = [
            (input_cell_gap * (i + 1), 0) + (0,) * (self.n_dims - 2) for i in range(n_in)
        ]
        output_cell_gap = shape[0] // (n_out + 1)
        self.output_indices = [
            (output_cell_gap * (i + 1), -1) + (0,) * (self.n_dims - 2) for i in range(n_out)
        ]

        # build the N-D cross-shaped convolutional filter for the activations net
        # For 2D, it looks like
        # [[0,1,0],
        #  [1,1,1],
        #  [0,1,0]]
        self.filter = np.array(self._build_filter(self.n_dims), dtype="float32")
        self.filter = tf.constant(
            tf.reshape(self.filter, [*self.filter.shape, 1, 1]), dtype="float32"
        )

    def step(self, inputs=None):
        # first, save the current step into the history
        self.activation_history.extend(tf.identity(self.activations_net))
        if inputs:
            # put new inputs into the input neurons
            for i, input_idx in enumerate(self.input_indices):
                tf.assign(self.activations_net[(0, *input_idx, 0)], inputs[i])

        print("Start of step:")
        self.print_net()
        # run update steps

        self.activations_net = tf.nn.convolution(self.activations_net, self.filter, "SAME")
        print("End of step:")
        self.print_net()

    def reset_activations(self):
        """Set all activations to 0, to "clear the net" for a fresh run of inputs
        Useful for when you don't want to carry over state from one input to the next
        """
        self.activations_net = tf.Variable(tf.zeros(tf.shape(self.multipliers_net)))
        self.activation_history = deque([], ACTIVATION_HISTORY_LEN)

    def _build_filter(self, dims, elem=None):
        if elem is None:
            elem = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        if dims != len(np.shape(elem)):
            if isinstance(elem, int):
                elem = [0, 1, 0] if elem == 0 else [1, 1, 1]
            # elem has less dimensions than it's supposed to, expand it
            for i, e in enumerate(elem):
                elem[i] = self._build_filter(dims - 1, e)
        return elem

    def print_net(self):
        print(tf.squeeze(self.activations_net).numpy())


net = DSNN((3, 3), 1, 1)
print("NET SHAPE", net.activations_net.shape)
print("FILTER SHAPE", net.filter.shape)

net.step([5])
net.step()
