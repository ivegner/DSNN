from keras.models import Model
from keras.layers import Input
from keras import backend as K
from keras.losses import mean_squared_error
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np

from itertools import zip_longest
from functools import reduce

from layer import SNNLayer


tf.enable_eager_execution()


class SNNModel():
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(inputs)
        return x

    @property
    def trainable_weights(self):
        trainable_weights = getattr(self, '_trainable_weights', [])
        if trainable_weights:
            return trainable_weights
        else:
            # gather the weights from the layers
            self._trainable_weights = []
            for x in self.layers:
                self._trainable_weights.extend(x.trainable_weights)
            return self._trainable_weights

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
                input("New X,Y")
                if np.any(sentinel == (x, y)):
                    raise ValueError("X and Y have different lengths!")

                # enter auto-grad context
                x = tf.constant(x, dtype="float32")
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x)
                    out = model(x)
                    if step_to_nonzero:
                        while np.isclose(out.numpy(), np.zeros_like(out), rtol=0.1).all():
                            print("activations\n", self.layers[0].activations)
                            input("Out: {}. Next step?".format(out.numpy()))
                            out = model(None)
                    print("FINAL OUT", out)
                    loss = tf.reduce_mean(tf.square(y - out))
                    print("LOSS", loss)
                    aggregate_loss += loss
                grads = tape.gradient(loss, self.trainable_weights[1:]) # 1: necessary to skip input matrix for now. TODO figure out why!!!!!!
                print("GRADIENT", grads)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights[1:],))
                del tape
            print("Epoch {} over - loss: {}".format(epoch, aggregate_loss))


if __name__ == "__main__":
    model = SNNModel([
        SNNLayer((3,3,1))
    ])
    NUM_EXAMPLES = 2000
    # training_inputs = tf.random.normal([NUM_EXAMPLES, 2])
    # noise = tf.random.normal([NUM_EXAMPLES, 2])
    # training_outputs = training_inputs * 3 + 2 + noise
    training_inputs = tf.random.normal([NUM_EXAMPLES, 1])
    noise = tf.random.normal([NUM_EXAMPLES, 1])
    training_outputs = training_inputs * 3 + 2 + noise


    model.train(training_inputs, training_outputs, epochs=1)

#     optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

#     print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

#     # Training loop
#     for i in range(300):
#     grads = grad(model, training_inputs, training_outputs)
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))
#     if i % 20 == 0:
#         print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

#     print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
#     print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))
