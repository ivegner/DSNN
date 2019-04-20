from keras.models import Model
from keras.layers import Input
from keras import backend as K
from keras.losses import mean_squared_error
import tensorflow as tf
import numpy as np

from itertools import zip_longest
from functools import reduce

from layer import SNNLayer


tf.enable_eager_execution()


def train(model, inputs, targets, epochs=10, step_to_nonzero=True):
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
            if np.any(sentinel == (x, y)):
                raise ValueError("X and Y have different lengths!")

            # enter auto-grad context
            x = tf.constant(x, dtype="float32")
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                out = model(x)
                # if step_to_nonzero:
                #     while np.isclose(out.numpy(), np.zeros_like(out), rtol=0.1):
                #         out = self.step()
                print("FINAL OUT", out)
                loss = tf.reduce_mean(tf.square(y - out))
                print("LOSS", loss)
                aggregate_loss += loss
            de_dm = tape.gradient(loss, model.trainable_weights)
            print("GRADIENT", de_dm)
            model.optimizer.apply_gradients(capped_gvs)
            del tape
            # self.multipliers_net.assign_sub(LEARNING_RATE * de_dm)
        print("Epoch {} over - loss: {}".format(epoch, aggregate_loss))


# class SNNModel(Model):
#     def __init__(self):
#         super(SNNModel, self).__init__()
#         self.layer1 = SNNLayer((3, 3, 2))

#     def __call__(self, input):
#         """Run the model."""
#         result = self.layer1(input)

#         return result

# model = SNNModel()

# data = np.array([[0, 1], [1, 2], [5, 6]])
# labels = np.array([1, 3, 11])

# train(model, data, labels, epochs=1)


class SNNModel(Model):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = SNNLayer((3, 3, 2))

    def call(self, inputs):
        return self.layer1(inputs)


# A toy dataset of points around 3 * x + 2

# # The loss function to be optimized
# def loss(model, inputs, targets):
#   error = model(inputs) - targets
#   return tf.reduce_mean(tf.square(error))

# def grad(model, inputs, targets):
#   with tf.GradientTape() as tape:
#     loss_value = loss(model, inputs, targets)
#   return tape.gradient(loss_value, model.trainable_weights)

# # Define:
# # 1. A model.
# # 2. Derivatives of a loss function with respect to model parameters.
# # 3. A strategy for updating the variables based on the derivatives.
if __name__ == "__main__":
    model = SNNModel()
    NUM_EXAMPLES = 2000
    training_inputs = tf.random.normal([NUM_EXAMPLES, 2])
    noise = tf.random.normal([NUM_EXAMPLES, 2])
    training_outputs = training_inputs * 3 + 2 + noise

    train(model, training_inputs, training_outputs, epochs=1)

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
