from keras import backend as K
from keras.layers import Layer
import tensorflow as tf

DECAY_FACTOR = 0.8

class SNNLayer(object):
    """Initialize a DSNN layer

    Args:
        shape (3D iterable): A tuple representing the shape of the Layer
            (length, width, N), where N = activation vector dimensionality
        output_type ("series"|"stack"): "series" for one output vector per timestep,
            "stack" to output the entire activation matrix
    """
    def __init__(self, shape, output_type="series"):
        if len(shape) != 3:
            raise Exception("Shape must be (length, width, N) tuple")
        if output_type not in ("series", "stack"):
            raise Exception("Output type must be one of (series|stack)")
        self.shape = shape
        self.output_type = output_type
        self.trainable_weights = []
        self.built = False

    def build(self, input_shape):
        if len(input_shape) == 3:
            # input is a "stack"
            self.input_type = "stack"
            assert self.shape == input_shape[:2]  # gotta match the activations matrix
            self.input_matrix = (
                self.add_weight(
                    name="input_matrix",
                    shape=(self.shape[0], self.shape[1], input_shape[-1], self.shape[-1]),
                    initializer=tf.random_normal_initializer(),
                    trainable=True,
                ),
            )
        elif len(input_shape) == 1:
            # input is a "series"
            self.input_type = "series"
            self.input_matrix = self.add_weight(  # simple 2D matrix, because input is a vector
                name="input_matrix",
                shape=(input_shape[0], self.shape[-1]),
                initializer=tf.random_normal_initializer(),
                trainable=True,
            )
        else:
            raise Exception(
                "Inter-layer invalid shape: {}. You should never see this error.".format(
                    input_shape
                )
            )
        self.activations = tf.zeros((self.shape[0], self.shape[1], 1, self.shape[2]))

        # 4 matrices for each neuron: [left, right, down, up]
        # index layer_weights[row,col,direction] => 2D matrix
        # self.layer_weights = [
        #     [
        #         [
        #             self.add_weight(
        #                 name="weight({},{},{}".format(row, column, direction),
        #                 shape=(input_shape[-1], input_shape[-1]),
        #                 initializer=tf.random_normal_initializer(),
        #                 trainable=True,
        #             )
        #             for direction in range(4)
        #         ]
        #         for column in range(self.shape[1])
        #     ]
        #     for row in range(self.shape[0])
        # ]
        # self.layer_weights = np.array(self.layer_weights, dtype=object)

        # {left: matrix, right: matrix, up: matrix, down: matrix}
        self.layer_weights = dict(
            (
                direction,
                self.add_weight(
                    name="layer_weights",
                    shape=(self.shape[0], self.shape[1], input_shape[-1], input_shape[-1]),
                    initializer=tf.random_normal_initializer(),
                    trainable=True,
                ),
            )
            for direction in ("left", "right", "up", "down")
        )
        self.built = True

    def __call__(self, x):
        """Perform a step of the DSNN

        Args:
            x (np.array/Tensor): Input for this timestep.
        """
        if x is not None:
            if not self.built:
                self.build(x.shape)
            if len(x.shape) == 1:
                x = tf.expand_dims(x, 0)
            _active = tf.matmul(x, self.input_matrix)

            if self.input_type == "series":
                # TODO: series is actually a special case of stack, so make it an input matrix
                # at layer initialization

                # x is actually a 1D vector, needs to be converted to a full activation
                # matrix, while preserving the gradient
                stack = self._series_to_stack(x, self.shape)
                stack = tf.expand_dims(stack, 2)
                _active = self.activations + stack

        else:
            _active = self.activations

        _OFFSET_DIRS = {
            "up": [-1,0], "down": [1,0], "left": [0,-1], "right":[0,1]
        }
        for direction, matrix in self.layer_weights.items():
            direction_result = tf.matmul(_active, matrix)
            direction_result # TODO: offset. So, e.g. for up, shift the matrix one row up
            self.activations +=

        # TODO: decay
        self.activations *= DECAY_FACTOR

        if self.output_type == "series":
            return self.activations[self.shape[0] - 1, self.shape[1] - 1]
        else:
            return self.activations

    # def compute_output_shape(self, input_shape):
    #     if self.output_type == "stack":
    #         return self.shape
    #     else:
    #         return self.shape[-1]  # dimensionality of the vector

    @staticmethod
    @tf.custom_gradient
    def _series_to_stack(x, shape):
        x = x[0]
        assert len(x.shape) == 1

        def grad(dy):
            # print("DY IN SERIES_TO_STACK", dy, dy.shape)
            return dy[0, 0]

        delta = (
            tf.sparse_tensor_to_dense(
                tf.SparseTensor(
                    [(0, 0, i) for i in range(shape[-1])],
                    [x[i].numpy() for i in range(shape[-1])],
                    shape,
                ),
                0,
            ),
        )[0]
        return delta, grad

    def add_weight(self, *args, **kwargs):
        weight = tf.get_variable(*args, **kwargs)
        if weight.trainable:
            self.trainable_weights.append(weight)
        return weight


"""
class CustomSparseTensor:
    def __init__(self, dense=None, shape=None):
        if dense is not None:
            assert len(dense.shape) == 3  # x,y,DIM
            self.shape = dense.shape
            vector_dim = self.shape[-1]
            self.indices = tf.where(
                tf.not_equal(x, tf.constant(tf.zeros((vector_dim,))))
            )
            self.values = tf.gather_nd(dense, self.indices)
        elif shape is not None:
            assert len(shape) == 3  # x,y,DIM
            self.shape = shape
            self.indices, self.values = [], []
b# build the inter-layer weights.
# 2 matrices for each neuron: up (aka depth--) and down (aka depth++)
# each up matrix is a transformation of R_{n_top_layer} -> R_{n_bottom_layer}
# each down matrix is a transformation of R_{n_bottom_layer} -> R_{n_top_layer}
# so, up shape is (dsnn_shape[0], dsnn_shape[1], (n_in +- d), (n_in +- d))

self.up_weights = [None] + [  # None = no up weights for first layer
    self.add_weight(
        name="up_weights_" + str(d),
        shape=(
            self.shape[0],
            self.shape[1],
            layer_dims[d],  # R_{top layer}
            layer_dims[d + 1],  # R_{bottom layer}
        ),
        initializer="normal", trainable=True
    )
    for d in range(self.shape[2] - 1)  # for each layer except the first
]

# None = no down weights for last layer
self.down_weights = [
    self.add_weight(
        name="down_weights_" + str(d),
        shape=(
            self.shape[0],
            self.shape[1],
            layer_dims[d],  # R_{top layer}
            layer_dims[d + 1],  # R_{bottom layer}
        ),
        initializer="normal", trainable=True
    )
    for d in range(self.shape[2] - 1)  # for each layer except the last
] + [None]




 """
