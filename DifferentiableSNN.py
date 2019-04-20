import tensorflow as tf
import numpy as np
import math

# from collections import deque
from itertools import zip_longest
from functools import reduce

tf.enable_eager_execution()

LEARNING_RATE = 0.1
# THRESHOLD_VALUE = 0.1
N_DEACTIVATION_STEPS = 2

# shape = (width, height, depth)
# n_in, n_out = number of inputs/outputs
class DSNN:
    def __init__(self, shape, n_in, n_out, do_threshold=False):
        """Initialize a DSNN

        Args:
            shape (3D tuple/list): A tuple representing the shape of the DSNN: (width, length, depth)
            n_in (int): Cardinality (length) of the input vector
            n_out (int): Cardinality(length) of the output vector
            do_threshold (bool, optional): Defaults to False. Whether to apply threshold function
        """
        self.shape = [d for d in shape if d > 1]  # no empty dimensions
        self.do_threshold = do_threshold

        self.n_dims = len(self.shape)
        if self.n_dims != 3:
            raise ValueError("DSNN dimensionality of more than 3 not yet supported")

        self.multipliers_net = tf.get_variable(
            "multipliers",
            shape=(1, *self.shape, 1),
            initializer=tf.random_normal_initializer(),
        )
        self.deactivation_masks = []

        self.activations_net = tf.zeros_like(self.multipliers_net)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

        if self.do_threshold:
            self.threshold = tf.Variable(
                tf.constant(0.5)
            )  # will be adjusted via gradient descent
            self.threshold_value = tf.Variable(
                tf.constant(1.0)
            )  # will be adjusted via gradient descent

        # first neuron of first layer for input, last neuron of last layer for output.
        self.input_index = [[0] * (n_dims + 2)]
        self.output_index = (
            (0,) + [self.shape[i] - 1 for i in range(self.n_dims)] + (0,)
        )

        # fix the multiplier of the input neurons at 1, just for niceness. They can't learn anyway.
        self.multipliers_net.assign(
            tf.scatter_nd_update(
                self.multipliers_net,
                self.input_indices,
                [1.0] * len(self.input_indices),
            )
        )

        # build the weights for the neuron connections

        # some math to figure out the length of the activation vector for each layer
        avg_change_per_layer = (n_out - n_in) / (self.shape[2] - 1)
        layer_dims = [
            n_in,
            *[
                math.floor(n_in + (d + 1) * avg_change_per_layer)
                for d in range(self.shape[2] - 2)
            ],
            n_out,
        ]

        # first, build the intra-layer weight matrix.
        # 4 matrices for each neuron: [left, right, down, up]
        # each matrix is a square matrix of side (n_in + d), where d=depth (if expansive, if compressive, n_in - d)
        # therefore, for each layer we need a shape (dsnn_shape[0], dsnn_shape[1], 4, (n_in +- d), (n_in +- d))
        # e.g. to get the "down" matrix for the C-th column in the R-th row in the L-th layer, we index
        # self.intralayer_weights[L][(R,C,2)]
        self.intralayer_weights = [
            tf.get_variable(
                "intralayer_weights_" + str(d),
                shape=(
                    1,
                    self.shape[0],
                    self.shape[1],
                    4,
                    layer_dims[d],
                    layer_dims[d],
                    1,
                ),
                initializer=tf.random_normal_initializer(),
            )
            for d in range(self.shape[2])  # for each layer
        ]

        # build the inter-layer weights.
        # 2 matrices for each neuron: up (aka depth--) and down (aka depth++)
        # each up matrix is a transformation of R_{n_top_layer} -> R_{n_bottom_layer}
        # each down matrix is a transformation of R_{n_bottom_layer} -> R_{n_top_layer}
        # so, up shape is (dsnn_shape[0], dsnn_shape[1], (n_in +- d), (n_in +- d))

        self.up_weights = [None] + [  # None = no up weights for first layer
            tf.get_variable(
                "up_weights_" + str(d),
                shape=(
                    1,
                    self.shape[0],
                    self.shape[1],
                    layer_dims[d],  # R_{top layer}
                    layer_dims[d + 1],  # R_{bottom layer}
                    1,
                ),
                initializer=tf.random_normal_initializer(),
            )
            for d in range(self.shape[2] - 1)  # for each layer except the first
        ]

        # None = no down weights for last layer
        self.down_weights = [
            tf.get_variable(
                "down_weights_" + str(d),
                shape=(
                    1,
                    self.shape[0],
                    self.shape[1],
                    layer_dims[d],  # R_{top layer}
                    layer_dims[d + 1],  # R_{bottom layer}
                    1,
                ),
                initializer=tf.random_normal_initializer(),
            )
            for d in range(self.shape[2] - 1)  # for each layer except the last
        ] + [None]

    def _threshold_func(self, x, threshold, threshold_value=1.0, steepness=10.0):
        """Thresholding function (sigmoid with coefficients)
         Transforms activations into a standard range in a differentiable manner

        Args:
            x (Tensor): The tensor/array to threshold
            threshold (float): Threshold, before which x~0 and after which, x~threshold_value
            threshold_value (float, optional): Defaults to 1.0. The maximum value of
                the threshold
            steepness (float, optional): Defaults to 10.0. The steepness of the intermediate
                part of the threshold function, between 0 and threshold_value

        Returns:
            Tensor: tensor of the same shape, thresholded to the provided parameters
        """
        active_mask = tf.abs(1.0 - self._get_zero_mask(x))  # invert
        t = threshold_value / (1.0 + tf.exp(-steepness * (x - threshold - 1.0 / np.e)))
        return (
            t * active_mask
        )  # set the zeroes back to 0, to avoid noisy fake activations

    def step(self, inputs=None):
        if inputs:
            # put new inputs into the input neurons
            # _active_net = tf.scatter_nd_update(tf.Variable(self.activations_net), self.input_indices, inputs)
            # for i, input_idx in enumerate(self.input_indices):
            #     tf.assign(self.activations_net[input_idx], inputs[i])
            # _active_net = tf.sparse_to_dense(
            #     [self.input_index], self.activations_net.shape, inputs, 1
            # )
            # _active_net = tf.multiply(
            #     tf.sparse_tensor_to_dense(
            #         tf.SparseTensor(
            #             [self.input_index],
            #             [inputs / _active_net[self.input_index]],
            #             self.activations_net.shape,
            #         ),
            #         1.0,
            #     ),
            #     self.activations_net,
            # )
            _active_net = tf.identity(self.activations_net)

        else:
            _active_net = self.activations_net

        self.print_net("Multipliers", self.multipliers_net)
        print("---\nStart of step:")
        self.print_net("Activations", _active_net)

        # build a boolean mask of the neurons active in the last 2 timesteps.
        # They will be deactivated at the end of the step
        active_mask = self._get_zero_mask(_active_net)
        self.deactivation_masks.append(active_mask)
        if len(self.deactivation_masks) > N_DEACTIVATION_STEPS:
            self.deactivation_masks.pop(0)
        # add up neighbors of every cell
        # this is equivalent to a convolution with a cross-shaped additive filter

        if self.do_threshold:
            # apply activation/threshold function
            print(
                "THRESHOLD: ",
                self.threshold.numpy(),
                " THRESHOLD VALUE: ",
                self.threshold_value.numpy(),
            )
            if self.threshold > self.threshold_value:
                raise Exception("Well... fuck.")

            self.print_net("BEFORE THRESHOLD", _active_net)
            _active_net = self._threshold_func(
                _active_net, self.threshold, threshold_value=self.threshold_value
            )
            self.print_net("AFTER THRESHOLD", _active_net)

        # multiply by the multipliers to get the final activations
        _active_net = _active_net * self.multipliers_net  # * self.threshold

        total_mask = reduce(lambda x, y: tf.multiply(x, y), self.deactivation_masks)
        self.activations_net = total_mask * _active_net
        print("---\nEnd of step:")
        self.print_net("Activations", _active_net)
        return tf.gather_nd(_active_net, self.output_indices)

    def _get_zero_mask(self, tensor):
        """Return a boolean mask where 1 is a 0 in the tensor.

        Args:
            tensor: Input tensor

        Returns:
            Tensor: Boolean mask of the zeroes in the tensor
        """
        active_mask = tf.where(tf.not_equal(tensor, tf.constant(0.0, dtype="float32")))
        active_mask = tf.sparse_tensor_to_dense(
            tf.SparseTensor(active_mask, tf.zeros((len(active_mask),)), tensor.shape),
            1.0,
        )
        return active_mask

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
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x)
                    out = self.step(x)
                    if step_to_nonzero:
                        while np.isclose(out.numpy(), np.zeros_like(out), rtol=0.1):
                            out = self.step()
                    # self.reset_activations()
                    print("FINAL OUT", out)
                    loss = tf.reduce_mean(tf.square(y - out))
                    print("LOSS", loss)
                    aggregate_loss += loss
                de_dm = tape.gradient(loss, self.multipliers_net)
                if self.do_threshold:
                    de_dt = tape.gradient(loss, self.threshold)
                    de_dtv = tape.gradient(loss, self.threshold_value)
                    print(
                        "THRESHOLD GRADIENT",
                        de_dt.numpy(),
                        "VALUE GRADIENT",
                        de_dtv.numpy(),
                    )
                self.print_net("GRADIENT", de_dm.numpy())

                capped_gvs = [
                    (tf.clip_by_value(grad, -1.0, 1.0), var)
                    for grad, var in [(de_dm, self.multipliers_net)]
                    + (
                        [(de_dt, self.threshold), (de_dtv, self.threshold_value)]
                        if self.do_threshold
                        else []
                    )
                ]

                self.optimizer.apply_gradients(capped_gvs)
                del tape
                # self.multipliers_net.assign_sub(LEARNING_RATE * de_dm)
            print("Epoch {} over - loss: {}".format(epoch, aggregate_loss))

    # def reset_activations(self):
    #     """Set all activations to 0, to "clear the net" for a fresh run of inputs
    #     Useful for when you don't want to carry over state from one input to the next
    #     """
    #     # self.activations_net.assign(tf.zeros_like(self.multipliers_net))
    #     self.activations_net = tf.zeros_like(self.multipliers_net)

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


net = DSNN((3, 3), 1, 1)
print("NET SHAPE", net.activations_net.shape)
print("FILTER SHAPE", net.filter.shape)

inputs = [[0.1], [0.2], [0.3], [0.4]]
targets = [[0.3], [0.5], [0.7], [0.9]]

net.train(inputs, targets, epochs=10)


print("----\n\n")
out1 = net.step([0.05]).numpy()
while out1 == 0:
    out1 = net.step().numpy()


out2 = net.step([0.25]).numpy()
while out2 == 0:
    out2 = net.step().numpy()

print(out1, " - expected - 0.2")
print(out2, " - expected - 0.6")
