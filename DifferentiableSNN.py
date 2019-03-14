import tensorflow as tf
import numpy as np

# from collections import deque
from itertools import zip_longest

tf.enable_eager_execution()

ACTIVATION_HISTORY_LEN = 10
LEARNING_RATE = 0.01

# shape = (width, height, depth)
# n_in, n_out = number of inputs/outputs
class DSNN:
    def __init__(self, shape, n_in, n_out):
        self.shape = [d for d in shape if d > 1]  # no empty dimensions
        self.n_dims = len(self.shape)
        self.multipliers_net = tf.get_variable(
            "multipliers", shape=(1, *self.shape, 1), initializer=tf.random_normal_initializer()
        )
        self.activations_net = tf.Variable(tf.zeros_like(self.multipliers_net))

        # first row for inputs, last row for outputs. TODO: input and output on different heights.
        input_cell_gap = shape[0] // (n_in + 1)
        self.input_indices = [
            (0, input_cell_gap * (i + 1), 0) + (0,) * (self.n_dims - 2) + (0,) for i in range(n_in)
        ]
        output_cell_gap = shape[0] // (n_out + 1)
        self.output_indices = [
            (0, output_cell_gap * (i + 1), self.shape[1] - 1) + (0,) * (self.n_dims - 2) + (0,)
            for i in range(n_out)
        ]

        # build the N-D cross-shaped convolutional filter for the activations net
        # e.g. for 2D, it looks like
        # [[0,1,0],
        #  [1,1,1],
        #  [0,1,0]]
        self.filter = np.array(self._build_filter(self.n_dims), dtype="float32")
        self.filter = tf.constant(
            tf.reshape(self.filter, [*self.filter.shape, 1, 1]), dtype="float32"
        )

    def step(self, inputs=None):
        # # first, save the current step into the history
        # self.activation_history.extend(tf.identity(self.activations_net))
        if inputs:
            # put new inputs into the input neurons
            self.activations_net = tf.scatter_nd_update(self.activations_net, self.input_indices, inputs)
            # for i, input_idx in enumerate(self.input_indices):
            #     tf.assign(self.activations_net[input_idx], inputs[i])

        print("Start of step:")
        self.print_net()

        # add up neighbors of every cell
        # this is equivalent to a convolution with a cross-shaped additive filter
        self.activations_net.assign(tf.nn.convolution(self.activations_net, self.filter, "SAME"))

        # apply activation/threshold function
        # TODO
        # multiply by the multipliers to get the final activations
        self.activations_net.assign(self.multipliers_net * self.activations_net)
        print("End of step:")
        self.print_net()
        return tf.gather_nd(self.activations_net, self.output_indices)

    def train(self, inputs, targets, epochs=10, step_to_nonzero=True):
        """
        `inputs`: iterable of inputs

        `targets`: iterable of corresponding labels

        `step_to_nonzero`: whether to backprop every step, or just the ones where the output is nonzero.
        If True (default), the training will feed in the input, then call `step()` until the output
        is nonzero, then backprop. If it is False, it will backprop every step - this probably means that you need to
        make input-output pairs for EVERY timestep.
        """
        try:
            if len(inputs) != len(targets):
                raise ValueError("X and Y have different length!")
        except TypeError:
            # no __len__ on x or y
            pass

        for epoch in range(epochs):
            sentinel = object()
            aggregate_loss = 0
            # Iterate while ensuring that x and y have the same lengths
            for (x, y) in zip_longest(inputs, targets, fillvalue=sentinel):
                if sentinel in (x, y):
                    raise ValueError("X and Y have different lengths!")

                # enter auto-grad context
                x = tf.constant(x, dtype="float32")
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    out = self.step(x)
                    if step_to_nonzero:
                        while out.numpy() == 0:
                            out = self.step()
                    print("FINAL OUT", out)
                    loss = tf.reduce_mean(tf.square(y - out))
                    print("LOSS", loss)
                    aggregate_loss += loss
                de_dm = tape.gradient(loss, self.multipliers_net)
                print("GRADIENT", de_dm, sep="\n")
                # self.optimizer.apply_gradients([(de_dm, self.multipliers_net)])
                self.multipliers_net = LEARNING_RATE * de_dm
            print("Epoch {} over - loss: {}".format(epoch, aggregate_loss))

    def reset_activations(self):
        """Set all activations to 0, to "clear the net" for a fresh run of inputs
        Useful for when you don't want to carry over state from one input to the next
        """
        self.activations_net.assign(tf.zeros_like(self.multipliers_net))
        # self.activation_history = deque([], ACTIVATION_HISTORY_LEN)

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
        print("Multipliers", tf.squeeze(self.multipliers_net).numpy(), sep="\n")
        print("Activations", tf.squeeze(self.activations_net).numpy(), sep="\n")
        print("\n")


net = DSNN((3, 3), 1, 1)
print("NET SHAPE", net.activations_net.shape)
print("FILTER SHAPE", net.filter.shape)

# print("Out: ", net.step([5]))
# print("Out: ", net.step())

inputs = [[1], [2], [3], [4]]
targets = [[2], [3], [4], [5]]

net.train(inputs, targets)
