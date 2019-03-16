import tensorflow as tf
import numpy as np

# from collections import deque
from itertools import zip_longest
from functools import reduce

tf.enable_eager_execution()

ACTIVATION_HISTORY_LEN = 10
LEARNING_RATE = 0.1
# THRESHOLD_VALUE = 0.1
N_DEACTIVATION_STEPS = 2

# shape = (width, height, depth)
# n_in, n_out = number of inputs/outputs
class DSNN:
    def __init__(self, shape, n_in, n_out):
        self.shape = [d for d in shape if d > 1]  # no empty dimensions
        self.n_dims = len(self.shape)
        self.multipliers_net = tf.get_variable(
            "multipliers", shape=(1, *self.shape, 1), initializer=tf.random_normal_initializer()
        )
        self.deactivation_masks = []

        self.activations_net = tf.zeros_like(self.multipliers_net)
        self.optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)

        # first row for inputs, last row for outputs. TODO: input and output on different heights.
        self.input_indices = [(0, i, 0) + (0,) * (self.n_dims - 2) + (0,) for i in range(n_in)]
        output_cell_gap = shape[0] // (n_out + 1)
        self.output_indices = [
            (0, self.shape[0] - (i + 1), self.shape[1] - 1) + (0,) * (self.n_dims - 2) + (0,)
            for i in range(n_out)
        ]

        # fix the multiplier of the input neurons at 1, just for niceness
        self.multipliers_net.assign(
            tf.scatter_nd_update(
                self.multipliers_net, self.input_indices, [1.0] * len(self.input_indices)
            )
        )

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
        if inputs:
            # put new inputs into the input neurons
            # _active_net = tf.scatter_nd_update(tf.Variable(self.activations_net), self.input_indices, inputs)
            # for i, input_idx in enumerate(self.input_indices):
            #     tf.assign(self.activations_net[input_idx], inputs[i])
            _active_net = tf.sparse_to_dense(
                self.input_indices, self.activations_net.shape, inputs, 0
            )

        else:
            _active_net = self.activations_net

        self.print_net("Multipliers", self.multipliers_net)
        print("---\nStart of step:")
        self.print_net("Activations", _active_net)

        # build a boolean mask of the neurons active in the last 2 timesteps.
        # They will be deactivated at the end of the step
        active_mask = tf.where(tf.not_equal(_active_net, tf.constant(0.0, dtype="float32")))
        active_mask = tf.sparse_tensor_to_dense(
            tf.SparseTensor(active_mask, tf.zeros((len(active_mask),)), _active_net.shape), 1.0
        )
        self.deactivation_masks.append(active_mask)
        if len(self.deactivation_masks) > N_DEACTIVATION_STEPS:
            self.deactivation_masks.pop(0)
        # add up neighbors of every cell
        # this is equivalent to a convolution with a cross-shaped additive filter
        _active_net = tf.nn.convolution(_active_net, self.filter, "SAME")

        # apply activation/threshold function
        # _active_net = tf.nn.relu(_active_net)
        # multiply by the multipliers to get the final activations
        _active_net = _active_net * self.multipliers_net

        total_mask = reduce(lambda x, y: tf.multiply(x, y), self.deactivation_masks)
        self.activations_net = total_mask * _active_net
        print("---\nEnd of step:")
        self.print_net("Activations", _active_net)
        return tf.gather_nd(_active_net, self.output_indices)

    def train(self, inputs, targets, epochs=10, step_to_nonzero=True):
        """Train the D-SNN on inputs and targets

        Arguments:
            inputs {iterable} -- iterable of inputs
            targets {iterable} -- iterable of corresponding labels/targets

        Keyword Arguments:
            epochs {int} -- Number of epochs to train for (default: {10})
            step_to_nonzero {bool} -- whether to backprop every step, or just the ones
            where the output is nonzero. If True (default), the training will feed in
            the input, then call `step()` until the output is nonzero,
            then backprop. If it is False, it will backprop every step -
            this probably means that you need to make input-output
            pairs for EVERY timestep. (default: {True})

        Raises:
            ValueError -- If the lengths of the inputs and targets iterables do not match
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
                    # self.reset_activations()
                    print("FINAL OUT", out)
                    loss = tf.reduce_mean(tf.square(y - out))
                    print("LOSS", loss)
                    aggregate_loss += loss
                de_dm = tape.gradient(loss, self.multipliers_net)
                # de_dm = tf.clip_by_global_norm([de_dm], 5.0)[0][0]
                print("GRADIENT", de_dm, sep="\n")
                self.optimizer.apply_gradients([(de_dm, self.multipliers_net)])
                # self.multipliers_net.assign_sub(LEARNING_RATE * de_dm)
            print("Epoch {} over - loss: {}".format(epoch, aggregate_loss))

    # def reset_activations(self):
    #     """Set all activations to 0, to "clear the net" for a fresh run of inputs
    #     Useful for when you don't want to carry over state from one input to the next
    #     """
    #     # self.activations_net.assign(tf.zeros_like(self.multipliers_net))
    #     self.activations_net = tf.zeros_like(self.multipliers_net)
    #     # self.activation_history = deque([], ACTIVATION_HISTORY_LEN)

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

    def print_net(self, label, net):
        print(label, tf.squeeze(net).numpy(), sep="\n")
        print("\n")


net = DSNN((3,3), 1, 1)
print("NET SHAPE", net.activations_net.shape)
print("FILTER SHAPE", net.filter.shape)

inputs = [[0.1], [0.2], [0.3], [0.4]]
targets = [[0.3], [0.5], [0.7], [0.9]]

net.train(inputs, targets, epochs=5)


print("----\n\n")
out = net.step([0.60]).numpy()
while (out == 0):
    out = net.step().numpy()
print(out, " - expected - 1.3")


out = net.step([0.25]).numpy()
while (out == 0):
    out = net.step().numpy()
print(out, " - expected - 0.6")
